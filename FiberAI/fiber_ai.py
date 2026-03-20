# ═══════════════════════════════════════════
# fiber_ai.py
# AI Optical Fiber Fault Detection
# Thejesh | ECE AIML | Veltech
# ═══════════════════════════════════════════

import joblib
import numpy as np
import time

print("=" * 50)
print("  AI Optical Fiber Fault Detection")
print("  Thejesh | ECE AIML | Veltech")
print("=" * 50)
print()

# ── Load AI Models ──
try:
    print("Loading AI models...")
    rf_model = joblib.load("rf_model.pkl")
    le       = joblib.load("label_encoder.pkl")
    scaler   = joblib.load("scaler.pkl")
    print("✅ Random Forest loaded!")
    print("✅ Label Encoder loaded!")
    print("✅ Scaler loaded!")

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Make sure these files are in FiberAI folder:")
    print("  rf_model.pkl")
    print("  label_encoder.pkl")
    print("  scaler.pkl")
    input("Press Enter to exit...")
    exit()

print()

# ── Fault information ──
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

led_map = {
    "Normal"           : "Green LED ON",
    "Bending"          : "Yellow LED ON",
    "Connector"        : "Yellow LED ON",
    "High_Attenuation" : "Red LED + Slow Beep",
    "Break"            : "Red LED + Fast Beep",
}

# ── Predict function ──
def predict_fault(sensor_val):
    output_power = (sensor_val / 4095) * 10.0
    attenuation  = (10.0 - output_power) / 5.0
    snr          = output_power * 3.5
    noise        = (4095 - sensor_val) / 4095.0
    input_power  = 10.0
    distance     = 0.0

    features = [[
        input_power,
        output_power,
        attenuation,
        noise,
        snr,
        distance
    ]]

    features_scaled = scaler.transform(features)
    pred_num        = rf_model.predict(features_scaled)
    fault_type      = le.inverse_transform(pred_num)[0]
    fault_distance  = dist_map[fault_type]
    fault_status    = status_map[fault_type]
    fault_led       = led_map[fault_type]

    return fault_type, fault_distance, fault_status, fault_led, output_power


# ── DEMO MODE — Type sensor values manually ──
print("=" * 50)
print("  DEMO MODE")
print("  Type sensor value to test AI!")
print("  Type exit to quit")
print("=" * 50)
print()
print("Sensor value guide:")
print("  4000 → Normal fiber")
print("  3000 → Connector loss")
print("  2000 → Bending fault")
print("  1200 → High Attenuation")
print("   200 → Fiber Break")
print()

while True:
    try:
        user_input = input("Enter sensor value (0 to 4095): ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        sensor_val = int(user_input)

        if sensor_val < 0 or sensor_val > 4095:
            print("Please enter value between 0 and 4095!")
            print()
            continue

        fault, dist, status, led, power = predict_fault(sensor_val)

        print()
        print("┌──────────────────────────────┐")
        print(f"│ Sensor Value : {sensor_val:<14}│")
        print(f"│ Output Power : {power:.2f} W         │")
        print(f"│ Fault Type   : {fault:<14}│")
        print(f"│ Distance     : {str(dist)+' km':<14}│")
        print(f"│ Status       : {status:<14}│")
        print(f"│ ESP32 LED    : {led:<14}│")
        print("└──────────────────────────────┘")
        print()

        print("LCD shows:")
        print("┌────────────────┐")
        if fault == "Normal":
            print("│FLT:Normal      │")
            print("│STATUS: NORMAL  │")
        else:
            line1 = f"FLT:{fault}"
            line2 = f"D:{dist}km {status}"
            print(f"│{line1[:16]:<16}│")
            print(f"│{line2[:16]:<16}│")
        print("└────────────────┘")
        print()

    except ValueError:
        print("Please enter a number only!")
        print()

    except KeyboardInterrupt:
        print()
        print("Stopped!")
        break
