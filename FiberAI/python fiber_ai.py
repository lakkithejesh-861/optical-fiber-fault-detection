import serial
import joblib
import numpy as np
import time

print("=" * 50)
print("  AI Optical Fiber Fault Detection")
print("  Python AI Model — Laptop Side")
print("=" * 50)
print()

print("Loading AI models...")
rf_model = joblib.load("rf_model.pkl")
le       = joblib.load("label_encoder.pkl")
scaler   = joblib.load("scaler.pkl")
print("All models loaded!")
print()

PORT = "COM3"
BAUD = 115200

print(f"Connecting to ESP32 on {PORT}...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2)
    print(f"ESP32 connected!")
except:
    print(f"Cannot connect to {PORT}")
    print("Change PORT to correct COM number!")
    exit()

dist_map = {
    "Normal"           : 0.0,
    "Bending"          : 2.4,
    "Break"            : 5.1,
    "Connector"        : 3.2,
    "High_Attenuation" : 4.8,
}

status_map = {
    "Normal"           : "NORMAL",
    "Bending"          : "WARNING",
    "Connector"        : "WARNING",
    "High_Attenuation" : "CRITICAL",
    "Break"            : "CRITICAL",
}

print()
print("Listening for ESP32 data...")
print("Press Ctrl+C to stop")
print("-" * 40)

while True:
    try:
        raw = ser.readline().decode("utf-8").strip()
        if not raw:
            continue

        if raw.startswith("SENSOR:"):
            sensor_val   = int(raw.replace("SENSOR:", ""))
            output_power = (sensor_val / 4095) * 10.0
            attenuation  = (10.0 - output_power) / 5.0
            snr          = output_power * 3.5
            noise        = (4095 - sensor_val) / 4095.0
            input_power  = 10.0
            distance     = 0.0

            features        = [[input_power, output_power,
                                attenuation, noise, snr, distance]]
            features_scaled = scaler.transform(features)
            pred_num        = rf_model.predict(features_scaled)
            fault_type      = le.inverse_transform(pred_num)[0]
            fault_distance  = dist_map[fault_type]
            fault_status    = status_map[fault_type]

            print(f"Sensor : {sensor_val}")
            print(f"Fault  : {fault_type}")
            print(f"Dist   : {fault_distance} km")
            print(f"Status : {fault_status}")
            print("-" * 40)

            result = f"{fault_type},{fault_distance},{fault_status}\n"
            ser.write(result.encode("utf-8"))

    except KeyboardInterrupt:
        print("Stopped!")
        ser.close()
        break
    except Exception as e:
        continue
