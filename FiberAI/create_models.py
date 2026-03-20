# ═══════════════════════════════════════════
# create_models.py
# Creates dataset and trains AI models
# Run this ONCE to generate model files!
# Thejesh | ECE AIML | Veltech
# ═══════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print("=" * 55)
print("  FIBER FAULT AI — Model Creator")
print("  Thejesh | ECE AIML | Veltech")
print("=" * 55)
print()

# ══════════════════════════════════════
# STEP 1 — CREATE DATASET
# ══════════════════════════════════════
print("STEP 1: Creating dataset...")
print()

np.random.seed(42)
N = 200  # 200 rows per fault type

data = {
    "input_power": np.random.uniform(8, 12, N * 5),

    "output_power": np.concatenate([
        np.random.uniform(7.5, 9.5, N),  # Normal
        np.random.uniform(5.0, 7.0, N),  # Bending
        np.random.uniform(0.1, 2.0, N),  # Break
        np.random.uniform(4.0, 6.0, N),  # Connector
        np.random.uniform(2.0, 4.5, N),  # High_Atten
    ]),

    "attenuation": np.concatenate([
        np.random.uniform(0.1, 0.5, N),  # Normal
        np.random.uniform(1.0, 2.5, N),  # Bending
        np.random.uniform(5.0, 9.0, N),  # Break
        np.random.uniform(2.0, 4.0, N),  # Connector
        np.random.uniform(3.5, 6.0, N),  # High_Atten
    ]),

    "noise": np.concatenate([
        np.random.uniform(0.01, 0.10, N),  # Normal
        np.random.uniform(0.10, 0.20, N),  # Bending
        np.random.uniform(0.30, 0.50, N),  # Break
        np.random.uniform(0.15, 0.25, N),  # Connector
        np.random.uniform(0.20, 0.35, N),  # High_Atten
    ]),

    "snr": np.concatenate([
        np.random.uniform(30, 40, N),  # Normal
        np.random.uniform(18, 28, N),  # Bending
        np.random.uniform(2,  10, N),  # Break
        np.random.uniform(15, 22, N),  # Connector
        np.random.uniform(8,  16, N),  # High_Atten
    ]),

    "distance": np.concatenate([
        np.zeros(N),                    # Normal = 0 km
        np.random.uniform(1, 4, N),    # Bending
        np.random.uniform(3, 8, N),    # Break
        np.random.uniform(1, 5, N),    # Connector
        np.random.uniform(2, 7, N),    # High_Atten
    ]),

    "fault": (
        ["Normal"]           * N +
        ["Bending"]          * N +
        ["Break"]            * N +
        ["Connector"]        * N +
        ["High_Attenuation"] * N
    )
}

df = pd.DataFrame(data)
df.to_csv("fiber_dataset.csv", index=False)

print("✅ Dataset created!")
print(f"   Total rows : {len(df)}")
print(f"   Columns    : {list(df.columns)}")
print()
print("   Rows per fault:")
for fault, count in df["fault"].value_counts().items():
    print(f"   {fault:<20} : {count} rows")
print()

# ══════════════════════════════════════
# STEP 2 — PREPARE DATA
# ══════════════════════════════════════
print("STEP 2: Preparing data...")
print()

# Separate inputs and answer
X = df[["input_power", "output_power",
        "attenuation", "noise", "snr", "distance"]]
y = df["fault"]

# Convert fault names to numbers
le = LabelEncoder()
y_numbers = le.fit_transform(y)

print("   Fault name → Number:")
for i, name in enumerate(le.classes_):
    print(f"   {name:<20} → {i}")
print()

# Normalize values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_numbers,
    test_size=0.2,
    random_state=42
)

print(f"   Training rows : {len(X_train)}")
print(f"   Testing rows  : {len(X_test)}")
print()

# ══════════════════════════════════════
# STEP 3 — TRAIN RANDOM FOREST
# ══════════════════════════════════════
print("STEP 3: Training Random Forest...")
print("   Please wait...")
print()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# TRAIN!
rf_model.fit(X_train, y_train)

# TEST accuracy
pred    = rf_model.predict(X_test)
acc_rf  = accuracy_score(y_test, pred)

print(f"   ✅ Training complete!")
print(f"   Accuracy = {acc_rf * 100:.1f}%")
print()

# ══════════════════════════════════════
# STEP 4 — SAVE ALL FILES
# ══════════════════════════════════════
print("STEP 4: Saving model files...")
print()

joblib.dump(rf_model, "rf_model.pkl")
print("   ✅ rf_model.pkl saved!")

joblib.dump(le, "label_encoder.pkl")
print("   ✅ label_encoder.pkl saved!")

joblib.dump(scaler, "scaler.pkl")
print("   ✅ scaler.pkl saved!")

print()

# ══════════════════════════════════════
# STEP 5 — VERIFY FILES EXIST
# ══════════════════════════════════════
print("STEP 5: Checking files...")
print()

files_to_check = [
    "rf_model.pkl",
    "label_encoder.pkl",
    "scaler.pkl",
    "fiber_dataset.csv"
]

all_good = True
for fname in files_to_check:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"   ✅ {fname:<25} ({size} bytes)")
    else:
        print(f"   ❌ {fname} NOT FOUND!")
        all_good = False

print()

if all_good:
    print("=" * 55)
    print("  ALL FILES CREATED SUCCESSFULLY!")
    print("=" * 55)
    print()
    print(f"  Random Forest Accuracy : {acc_rf*100:.1f}%")
    print()
    print("  Now run fiber_ai.py to test!")
    print("  Type: python fiber_ai.py")
else:
    print("❌ Some files missing! Run again!")

print()
input("Press Enter to close...")
