import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Створення нейромережі
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),  # Вхідний шар
    Dense(32, activation='relu'),                     # Прихований шар
    Dense(1, activation='sigmoid')                    # Вихідний шар (бінарна класифікація)
])

# Компільовуємо модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Виведення структури моделі
model.summary()