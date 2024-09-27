import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Генерация данных
np.random.seed(42)
# Нормальные данные
X_normal = np.random.randn(100, 2)
# Аномальные данные
X_anomalies = np.random.uniform(low=-6, high=6, size=(10, 2))

# Объединение данных
X = np.concatenate((X_normal, X_anomalies), axis=0)

# Модель Isolation Forest
model = IsolationForest(contamination=0.3)  # Указываем долю аномалий
model.fit(X)
# Прогнозирование
y_pred = model.predict(X)

# Визуализация
plt.figure(figsize=(10, 6))
# Нормальные точки
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], label='Нормальные данные', color='blue')
# Аномальные точки
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], label='Аномальные данные', color='red')
plt.title('Обнаружение аномалий с помощью Isolation Forest')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.legend()
plt.grid()
plt.show()
