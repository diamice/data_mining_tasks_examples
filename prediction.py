import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Генерация полиномиальных данных
data_size = 200
X = np.linspace(-3, 3, data_size)  # Входные данные
y = 2 * X**2 + 3 * X + 5 + np.random.normal(scale=3, size=data_size)  # Полиномиальная функция с шумом

# Изменение формы данных для Keras
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1,)))  # Входной слой с 64 нейронами
model.add(Dense(64, activation='relu'))  # Скрытый слой
model.add(Dense(1))  # Выходной слой

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X_train, y_train, epochs=100, verbose=1)

# Прогнозирование
y_pred = model.predict(X_test)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Исходные данные', color='blue', alpha=0.5)
plt.scatter(X_test, y_test, label='Тестовые данные', color='green')
plt.scatter(X_test, y_pred, label='Прогнозированные данные', color='red')
plt.title('Прогнозирование полиномиальной функции с помощью Keras')
plt.xlabel('Входные данные')
plt.ylabel('Целевые значения')
plt.legend()
plt.grid()
plt.show()
