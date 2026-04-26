"""
Train LSTM model on heart disease tabular data.
Saves: lstm_model.h5, tokenizer.pkl
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ── Load & preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

# Encode categoricals
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save "tokenizer" (label encoders dict) with pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("tokenizer.pkl saved.")

# Features / target
X = df.drop("HeartDisease", axis=1).values.astype(np.float32)
y = df["HeartDisease"].values

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler for inference
with open("lstm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for LSTM: (samples, timesteps=1, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Build LSTM ─────────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ── Train 20 epochs ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

model.save("lstm_model.h5")
print("lstm_model.h5 saved.")
