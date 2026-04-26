"""
Train 1D CNN model on heart disease tabular data.
Saves: cnn_model.h5
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# ── Load & preprocess ──────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop("HeartDisease", axis=1).values.astype(np.float32)
y = df["HeartDisease"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler for inference
with open("cnn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for Conv1D: (samples, features, 1)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Build 1D CNN ───────────────────────────────────────────────────────────────
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation="relu"),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ── Train 30 epochs ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

model.save("cnn_model.h5")
print("cnn_model.h5 saved.")
