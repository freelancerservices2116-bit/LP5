import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("Boston_housing.csv")
df.columns = df.columns.str.replace('"', '')

X = df.drop('MEDV', axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)   # Linear output (regression)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

loss, mae = model.evaluate(X_test, y_test)

print("Test MSE:", loss)
print("Test MAE:", mae)

y_pred = model.predict(X_test).flatten()

print("\nSample Predictions:")
for actual, pred in zip(y_test[:5], y_pred[:5]):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")

plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.show()