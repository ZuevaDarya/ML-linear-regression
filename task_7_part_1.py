import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data_table = pd.read_csv('input.csv', delimiter=',', decimal='.', index_col='id')
data_values = data_table.values

#Выбираем значения из 1-го столбца таблицы
X = data_values[:, 0].reshape(-1, 1)
#Выбираем значения из 2-го столбца таблицы (отклик)
Y = data_values[:, 1].reshape(-1, 1)

X_mean = np.average(X)
Y_mean = np.average(Y)
print(f'X выборочное среднее: {X_mean}') #11.4
print(f'Y выборочное среднее: {Y_mean}') #25.0

#Обучение модели
model = LinearRegression().fit(X, Y)

# y = t0 + t1x1 
#тетта с ноликом
t_0 = model.intercept_
t_1 = model.coef_
print(f'Тетта с ноликом t_0: {round(float(t_0), 2)}') #5.68
print(f'Тетта с 1 t_01: {round(float(t_1), 2)}') #1.69

#R^2 статистика (точность модели)
R_2 = model.score(X, Y)
print(f'R^2 статистика: {round(float(R_2), 2)}') #0.79

