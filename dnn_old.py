import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from tensorflow import keras
from datetime import datetime
from mpldatacursor import datacursor


def create_test_matrix(x_range: list = (0, 100), y_range: list = (0, 100), resolution=100):
    # создаем массив нужной нам размерности
    array = np.zeros([resolution, resolution, 2])
    # считаем шаг для X
    x_step = abs((x_range[1] - x_range[0]) / resolution)
    # считаем шаг для Y
    y_step = abs((y_range[1] - y_range[0]) / resolution)
    for i in range(resolution):
        for j in range(resolution):
            array[i, j] = np.array([x_range[0] + x_step * i, y_range[0] + y_step * j])
    array = np.flipud(array)
    return array


# загружаем данные
def load_data(path):
    file = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    # print(pd.DataFrame(file.loc[:, ['Широта', 'Долгота', 'Расстояние до набережной']]).to_numpy().max(axis=0))
    return pd.DataFrame(file.loc[:, ['Широта', 'Долгота', 'Расстояние до набережной']]).to_numpy()


def load_river(path):
    file = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(file.loc[:, ['Широта', 'Долгота']]).to_numpy()


# нормализация данных
def scale_data(fit=None, transform=None):
    scaler = QuantileTransformer(output_distribution='uniform')
    scaler.fit(fit)
    return scaler.transform(transform)


# Отрисовка данных
def show_data(buildings=None, river=None, predict_matrix=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(buildings[:, 1], buildings[:, 0], color='black', s=2)
    ax[0].scatter(river[:, 1], river[:, 0], color='red')
    ax[0].set_title("After scaling")
    # im = ax[1].contourf(predict_matrix)
    for i in range(predict_matrix.shape[1]):
        for j in range(predict_matrix.shape[1]):
            if 1500 < predict_matrix[i, j] < 1700:
                predict_matrix[i, j] = 1.0
            elif 1000 < predict_matrix[i, j] < 1500:
                predict_matrix[i, j] = 2.0
            elif 500 < predict_matrix[i, j] < 1000:
                predict_matrix[i, j] = 3.0
            elif 200 < predict_matrix[i, j] < 500:
                predict_matrix[i, j] = 4.0
            elif predict_matrix[i, j] < 200:
                predict_matrix[i, j] = 6.0
            else:
                predict_matrix[i, j] = 0.0

    im = ax[1].imshow(predict_matrix, interpolation='nearest')
    fig.colorbar(im, extend='both')
    ax[1].set_title("predicted data")
    datacursor(hover=False, bbox=dict(alpha=1, fc='w'), formatter='{z:.06g}'.format)
    plt.tight_layout()
    plt.show()


# создаем нейронную сеть
def create_model(input_size=None):
    assert input_size is not None
    model = keras.Sequential()
    # первый слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(256, input_dim=input_size, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('relu'))
    # второй слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('relu'))
    # третий слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # четвертый слой ---------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # четвертый слой ---------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # model.add(keras.layers.Dropout(0.2))
    # четвертый слой ---------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # четвертый слой ---------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(1))
    # компилируем нейронную сеть
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def model_fit(model=None, X_train_norm=None, Y_train=None, batch_size=1, epochs=1, device='cpu'):
    assert model is not None and X_train_norm is not None and Y_train is not None
    # TensorBoard ------------------------------------------------------------------------------------------------------
    callback_tb = keras.callbacks.TensorBoard(log_dir='dnn_log', histogram_freq=1, write_graph=True, write_images=True)
    # Келбек сохраняющий веса модели -----------------------------------------------------------------------------------
    # Добавим эпоху в имя файла (uses `str.format`)
    date = datetime.today().strftime("%m.%d.%Y %H.%M")
    path = 'weights {}'.format(date) + "/cp-{epoch:04d}.ckpt"
    callback_as = keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True, verbose=1, period=2)
    # Сохраним веса 1 эпохи, используя формат `checkpoint_path` format
    model.save_weights(path.format(epoch=0))
    # Келбек снижающий скорость обучения -------------------------------------------------------------------------------
    callback_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0005)
    tf.device('/{}:0'.format(device))
    # заполнитель (обучение)
    model.fit(X_train_norm, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1,
              callbacks=[callback_tb, callback_as, callback_lr])
    return model


def get_last_dir():
    list_dir = pd.DataFrame(os.listdir('.'))
    list_dir = list_dir[list_dir[0].str.contains(r'weight')].sort_values(by=[0])
    list_dir = list_dir.reset_index(drop=True)
    return str(list_dir.iloc[-1, 0])


def main():
    path = 'D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/обработанные данные/' \
           'ЗДАНИЯ.csv'
    data = load_data(path)
    # разбиваем данные на 2 выборки: тренировочную и тестовую
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    print('Размер датасета: {}\n'
          'Размер тренировочной выборки: {}\n'
          'Размер тестовой выборки: {}\n'.format(data.shape, train_set.shape, test_set.shape))
    # разделяем на X и Y обе выборки. Где X - входы. Y - выходы
    X_train = train_set[:, :2]
    Y_train = train_set[:, 2].reshape(-1, 1)
    X_test = test_set[:, :2]
    Y_test = test_set[:, 2].reshape(-1, 1)
    print('Тренировочная выборка:\n Входы: {}\n Выходы: {}\n'.format(X_train.shape, Y_train.shape))
    print('Тестовая выборка:\n Входы: {}\n Выходы: {}\n'.format(X_test.shape, Y_test.shape))
    # нормализация данных
    X_train_norm = scale_data(X_train, X_train)
    X_test_norm = scale_data(X_train, X_test)

    t = input('Режим\n1 | Обучение\n2 | Загрузка\n')
    if t == '1':
        # создаем нейронную сеть
        model = create_model(X_train_norm.shape[1])
        # обучаем модель
        model = model_fit(model, X_train_norm, Y_train, 40, 1000)
    else:
        t = input('Режим\n1 | Загрузка весов\n2 | Загрузка модели\n')
        if t == '1':
            model = create_model(X_train_norm.shape[1])
            lastest = tf.train.latest_checkpoint(get_last_dir())  # weights 04.18.2020 21.45
            print(lastest)
            model.load_weights(lastest)
        else:
            model = keras.models.load_model('last training model.h5')
        t = input('Продолжить обучение?\n1 | Да\n2 | Нет\n')
        if t == '1':
            # обучаем модель
            model = model_fit(model, X_train_norm, Y_train, 40, 1000)
    model.summary()
    # оценка сети
    mse, mae = model.evaluate(X_test_norm, Y_test, verbose=0)
    print("Test mae: {}".format(mae))

    # визуализация результатов обучения нейронной сети
    # получем диапазон координат по X и Y
    x_range = [X_test.min(axis=0)[0], X_test.max(axis=0)[0]]
    y_range = [X_test.min(axis=0)[1], X_test.max(axis=0)[1]]
    print(x_range, y_range)
    # создаем матрицу с координатами в ячейках
    resolution = 500
    xy_matrix = create_test_matrix(x_range, y_range, resolution)
    # нормализируем эти значения (построчно)
    for i in range(xy_matrix.shape[0]):
        xy_matrix[i] = scale_data(X_train, xy_matrix[i])
    # создаем матрицу той же размерности, но с ответом НС в ячейках
    predict_matrix = np.zeros([resolution, resolution])
    # получем ответ от НС и заполняем матрицу
    for i in range(xy_matrix.shape[0]):
        predict_matrix[i] = model.predict(xy_matrix[i]).reshape(-1)
    # считываем файл с данными о расположении берегов
    river_data = load_river('D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/'
                            'обработанные данные/НАБЕРЕЖНЫЕ.csv')
    # отрисовываем полученый результат
    show_data(X_train, river_data, predict_matrix)

    # предсказание
    pred = model.predict(X_test_norm)
    summ = 0
    for i in range(Y_test.shape[0]):
        # print(((pred[i][0] - Y_test[i]) / Y_test[i]) * 100)
        summ += ((pred[i][0] - Y_test[i]) / Y_test[i]) * 100
    print('{} %'.format(summ / Y_test.shape[0]))

    pred = model.predict(X_train_norm)
    summ = 0
    for i in range(Y_train.shape[0]):
        summ += ((pred[i][0] - Y_train[i]) / Y_train[i]) * 100
    print('{} %'.format(summ / Y_train.shape[0]))

    t = input('Сохранить модель?\n1 | Да\n2 | Нет\n')
    if t == '1':
        # сохранение модели
        date = datetime.today().strftime("%m.%d.%Y %H.%M")
        model.save('model {date}.h5'.format(date=date))

if __name__ == '__main__':
    main()
