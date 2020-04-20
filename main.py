import pandas as pd
import numpy as np
from datetime import datetime
from dataloader import DataLoader
from transformer import Transformer
from dnn import Dnn


def main():
    # создаем имя папки, куда будут сохраняться данные текущей сессии
    date: str = datetime.today().strftime("%d.%m.%Y %H.%M")
    # считываем данные и разбиваем данные на 2 выборки: тренировочную и тестовую.
    # X - данные на вход. Y - данные на выход
    dataLoader = DataLoader()
    path = 'D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/обработанные данные/ЗДАНИЯ.csv'
    labels = ['Широта', 'Долгота', 'Расстояние до набережной']
    # разбиваем данные на 2 выборки: тренировочную и тестовую. X - данные на вход. Y - данные на выход
    t = str(input("Использовать существующую нормализацию?\n1 | Да\n2 | Нет"))
    if t == '1':
        fit_path = str(input())
    else:
        fit_path = None
    (X_train, Y_train), (X_test, Y_test) = dataLoader.load(path=path, labels=labels, test_size=0.2,
                                                           random_state=42, fit_path=fit_path)
    # сохраняем данные он нормализации в папку
    dataLoader.tr.save_transformer(date)
    # создаем экземпляр класса DNN
    dnn = Dnn()
    t = str(input('Выберите действие:\n1 | Обучить\n2 | Загрузить веса\n3 | Загрузть модель'))
    if t == '1':
        layers = [256, 128, 128, 128, 128, 128]
        activations = ['relu', 'relu', 'elu', 'elu', 'elu', 'elu']
        dnn.create(inputs=X_train.shape[1], outputs=Y_train.shape[1], layers=layers, activations=activations)
        dnn.fit(X_train_norm=X_train, Y_train=Y_train, batch_size=40, epochs=100, dir_name=date)
    elif t == '2':
        layers = [256, 128, 128, 128, 128, 128]
        activations = ['relu', 'relu', 'elu', 'elu', 'elu', 'elu']
        dnn.create(inputs=X_train.shape[1], outputs=Y_train.shape[1], layers=layers, activations=activations)
        dnn.load_weights()  # 'weights 04.18.2020 21.45')
    elif t == '3':
        dnn.load_model()
    else:
        return
    # делаем оценку полученой сети
    mse, mae = dnn.evaluate(X_test, Y_test)
    print('mae: {}m.'.format(mae))
    print('mae: {}%.'.format(mae))
    t = input('Сохранить модель?\n1 | Да\n2 | Нет\n')
    if t == '1':
        # сохранение модели
        dnn.save_model(date)



    # предсказание
    pred = dnn.predict(X_test)
    summ = 0
    for i in range(Y_test.shape[0]):
        # print(((pred[i][0] - Y_test[i]) / Y_test[i]) * 100)
        summ += ((pred[i][0] - Y_test[i]) / Y_test[i]) * 100
    print('{} %'.format(summ / Y_test.shape[0]))

    pred = dnn.predict(X_train)
    summ = 0
    for i in range(Y_train.shape[0]):
        summ += ((pred[i][0] - Y_train[i]) / Y_train[i]) * 100
    print('{} %'.format(summ / Y_train.shape[0]))


if __name__ == "__main__":
    main()
