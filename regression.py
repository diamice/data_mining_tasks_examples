import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Создание данных
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 образцов, 3 независимые переменные
true_weights = np.array([3, 5, -2])  # истинные веса
y = X @ true_weights + np.random.randn(100) * 0.5  # зависимая переменная с шумом

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели
models = {
    'Линейная регрессия': LinearRegression(),
    'Регрессия Риджа': Ridge(alpha=1.0),
    'Регрессия Лассо': Lasso(alpha=0.1)
}

# Обучение моделей и оценка
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name} Среднеквадратичная ошибка: {mse:.2f}')
    print(f'{name} Коэффициенты: {model.coef_}\n')

# Визуализация коэффициентов
coefficients = pd.DataFrame(
    {name: model.coef_ for name, model in models.items()},
    index=[f'Признак {i+1}' for i in range(X.shape[1])]
)

coefficients.plot(kind='bar')
plt.title('Коэффициенты моделей регрессии')
plt.ylabel('Значение коэффициента')
plt.xticks(rotation=0)
plt.show()
