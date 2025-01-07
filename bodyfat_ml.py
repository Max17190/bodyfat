import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/Users/maxloffgren/Documents/Python files/Lifting/bodyfat.csv")

# Filter essential columns from data
df_ess = df[['BodyFat', 'Age', 'Weight', 'Height', 'Density', 'Abdomen']]
df_essentials = df_ess.copy()

# Filter outliers (IRQ)
Q1 = df_essentials.quantile(0.25)
Q3 = df_essentials.quantile(0.75)
IQR = Q3 - Q1
df_filtered = df_essentials[~((df_essentials < (Q1 - 1.5 * IQR)) | (df_essentials > (Q3 + 1.5 * IQR))).any(axis=1)]

# Define inputs and target
x_axis = df_essentials[['Age', 'Weight', 'Height', 'Density', 'Abdomen']]
y_axis = df_essentials['BodyFat']

# Assign train & test and normalize data
X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis, test_size=0.2, random_state=20)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model archi
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model w/ Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

# Training
history = model.fit(X_train_scaled, y_train, epochs=75, batch_size=6, validation_split=0.3)

# Predictions vs actual
y_test_pred_nn = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_test_pred_nn)
mse = mean_squared_error(y_test, y_test_pred_nn)
r2 = r2_score(y_test, y_test_pred_nn)
rmse = np.sqrt(mse)  

print("Neural Network Model Performance:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2: {r2}")
print(f"RMSE: {rmse}")

# Training loss and validation loss

"""
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
"""
