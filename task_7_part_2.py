import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

candy_table = pd.read_csv('candy-data.csv', delimiter=',', decimal='.', index_col='competitorname')
candy_data = pd.read_csv('candy-data.csv')
candy_data_values = candy_data.values

winpercent_col_index = 12
not_include_candies = ['One quarter', 'Peanut butter M&Ms']

#Исключаем конфеты из тренировочного списка
data_for_train = np.array(list(filter(lambda x: x[0] not in not_include_candies, candy_data_values)))

# #Выбираем значения предикторов Х и откликов У
X = data_for_train[:, 1:winpercent_col_index]
Y = data_for_train[:, winpercent_col_index].reshape(-1, 1)     

#Обучение модели
model = LinearRegression().fit(X, Y)
print(model.score(X, Y))
#Предсказывание значения
def predictScoreByName(name, data):
    #Выбираем нужную строку с данными (исключая первое текстовое поле и поле отклика)
    for row in data:
        if row[:1] == name: 
            current_row = [row[1:winpercent_col_index]]

    predict_value = model.predict(current_row)[0][0]
    return predict_value

for candy in not_include_candies:
    predict_value = predictScoreByName(candy, candy_data_values)
    print(f'Предсказанное значение для {candy}: {round(float(predict_value), 3)}')


new_candy_parameters = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0.505, 0.618]
predict_value = model.predict([new_candy_parameters])
print(f'Предсказанное значение для НОВОЙ КОНФЕТЫ: {round(float(predict_value), 3)}')

