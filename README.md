# DEEP-LEARNING
import tensorflow as tf
from tensorflow.keras import layers, models

# Buat model DNN (Deep Neural Network)
model = models.Sequential([
    layers.Input(shape=(X_train_processed.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output regresi
])

# Kompilasi model DNN (Deep Neural Network)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Ringkasan model (opsional)
model.summary()

# Latih model DNN (Deep Neural Network)
history = model.fit(
    X_train_processed, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1
)


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Prediksi terhadap data test 
y_pred = model.predict(X_test_processed).flatten()

# Hitung MAE dan RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Tampilkan hasil evaluasi
print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📉 Root Mean Squared Error (RMSE): {rmse:.2f}")

import matplotlib.pyplot as plt

# 1. Scatter plot: Prediksi vs Aktual
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # garis ideal
plt.xlabel('Harga Aktual')
plt.ylabel('Harga Prediksi')
plt.title('Prediksi vs Harga Aktual')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Grafik Loss Training vs Validation
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


