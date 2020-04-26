import pandas as pd
import numpy as np
from datetime import datetime
from dataloader import DataLoader
from Dnn import DnnCenter, DnnRegions, DnnPromenade
from transformer import Transformer

import matplotlib.pyplot as plt
from mpldatacursor import datacursor

def show_data(matrix):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    # im = ax[1].contourf(predict_matrix)
    # for i in range(matrix.shape[1]):
    #     for j in range(matrix.shape[1]):
    #         if 6000 < matrix[i, j]:
    #             matrix[i, j] = 0.0
    #         elif 3000 < matrix[i, j] < 6000:
    #             matrix[i, j] = 1.0
    #         elif 1500 < matrix[i, j] < 3000:
    #             matrix[i, j] = 2.0
    #         elif 1000 < matrix[i, j] < 1500:
    #             matrix[i, j] = 3.0
    #         elif 500 < matrix[i, j] < 1000:
    #             matrix[i, j] = 4.0
    #         elif matrix[i, j] < 500:
    #             matrix[i, j] = 5.0
    im = ax.imshow(matrix, interpolation='nearest')
    fig.colorbar(im, extend='both')
    ax.set_title("predicted data")
    datacursor(hover=False, bbox=dict(alpha=1, fc='w'), formatter='{z:.06g}'.format)
    plt.tight_layout()
    plt.show()


def main():
    # создаем имя папки, куда будут сохраняться данные текущей сессии
    # date: str = datetime.today().strftime("%d.%m.%Y %H.%M")
    # считываем данные и разбиваем данные на 2 выборки: тренировочную и тестовую.
    # X - данные на вход. Y - данные на выход
    dataLoader = DataLoader()
    path = 'D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/обработанные данные/ЗДАНИЯ.csv'
    labels = ['Широта', 'Долгота', 'Расстояние до набережной']
    # разбиваем данные на 2 выборки: тренировочную и тестовую. X - данные на вход. Y - данные на выход
    (X_train, Y_train), (X_test, Y_test) = dataLoader.load(path=path, labels=labels, test_size=0.2)

    # создаем экземпляр класса DNN
    dnn = DnnCenter()
    t = str(input('Выберите действие:\n1 | Обучить\n2 | Загрузить веса\n3 | Загрузть модель'))
    if t == '1':
        dnn.create()
        dnn.fit(X_train, Y_train)
    elif t == '2':
        dnn.create()
        dnn.load_weights()  # 'weights 04.18.2020 21.45')
        dnn.load_x_transformer()
    elif t == '3':
        dnn.load_model()
        dnn.load_x_transformer()
    else:
        return
    # делаем оценку полученой сети
    evaluate = dnn.evaluate(X_test, Y_test)
    print('------ data test ------')
    print('evaluate: {}'.format(evaluate))
    evaluate = dnn.evaluate(X_train, Y_train)
    print('------ data train -----')
    print('evaluate: {}'.format(evaluate))

    step = .005
    # Получаем диапазон координат по X и Y
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    # Генерируем матрицу с шагом сетки 'step'
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    # запоминаем размерность матрицы для дальнейшего отображения
    shape = xx.shape
    # получаем классификацию
    # np.c_[] - склеивает столбцы в одну матрицу
    # np.ravel() - преобразовывает массив в одномерный
    Z = dnn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Преобразовываем результат в матрицу
    Z = Z.reshape(shape)
    plt.figure()
    plt.pcolormesh(yy, xx, Z)  # , cmap=cmap_light)

    river = pd.read_csv('D:/Documents/учёба/КНТ/4 курс 2 семестр/'
                        'dataset недвижимость СПБ/обработанные данные/ШКОЛЫ.csv',
                        sep=';', encoding='utf-8-sig')
    river = pd.DataFrame(river.loc[:, ['Широта', 'Долгота']]).to_numpy()
    plt.scatter(river[:, 1], river[:, 0], color='white', s=1, alpha=0.5)
    plt.show()

    t = input('Сохранить модель?\n1 | Да\n2 | Нет\n')
    if t == '1':
        # сохранение модели
        dnn.save_model()


if __name__ == "__main__":
    main()
