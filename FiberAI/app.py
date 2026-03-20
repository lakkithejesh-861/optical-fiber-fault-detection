from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("Loading AI models...")
rf_model = joblib.load("rf_model.pkl")
le       = joblib.load("label_encoder.pkl")
scaler   = joblib.load("scaler.pkl")
print("All models loaded!")

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

icon_map = {
    "Normal"           : "✅",
    "Bending"          : "⚠️",
    "Connector"        : "⚠️",
    "High_Attenuation" : "🔴",
    "Break"            : "💥",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data       = request.get_json()
        sensor_val = int(data["sensor"])

        output_power = (sensor_val / 4095) * 10.0
        attenuation  = (10.0 - output_power) / 5.0
        snr          = output_power * 3.5
        noise        = (4095 - sensor_val) / 4095.0

        features = [[10.0, output_power, attenuation, noise, snr, 0.0]]

        features_scaled = scaler.transform(features)
        pred_num        = rf_model.predict(features_scaled)
        fault_type      = le.inverse_transform(pred_num)[0]

        return jsonify({
            "success"      : True,
            "sensor"       : sensor_val,
            "fault"        : fault_type,
            "distance"     : dist_map[fault_type],
            "status"       : status_map[fault_type],
            "icon"         : icon_map[fault_type],
            "output_power" : round(output_power, 2),
            "attenuation"  : round(attenuation, 3),
            "snr"          : round(snr, 2),
            "noise"        : round(noise, 3),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    print()
    print("=" * 45)
    print("  FiberAI Flask Server Starting!")
    print("  Open browser: http://localhost:5000")
    print("=" * 45)
    app.run(debug=True, port=5000)