import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tensorflow import keras


# загружаем данные
def load_data(path):
    file = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    return pd.DataFrame(file.loc[:, ['Широта', 'Долгота', 'Расстояние до набережной']]).to_numpy()


# нормализация данных
def scale_data(fit=None, transform=None):
    scaler = QuantileTransformer(output_distribution='uniform')
    scaler.fit(fit)
    return scaler.transform(transform)


# Отрисовка данных
def show_data(data=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.scatter(data[:, 0], data[:, 1], color='blue')
    ax.set_title("After scaling (data)")
    plt.tight_layout()
    plt.show()


# создаем нейронную сеть
def create_model(input_size=None):
    assert input_size is not None
    model = keras.Sequential()
    # первый слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, input_dim=input_size, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # второй слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # третий слой ------------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('elu'))
    # четвертый слой ---------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(128, use_bias=True, kernel_initializer='glorot_uniform'))
    model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
    model.add(keras.layers.Activation('tanh'))
    # model.add(keras.layers.Dropout(0.2))
    # выходной слой ----------------------------------------------------------------------------------------------------
    model.add(keras.layers.Dense(1))
    # компилируем нейронную сеть
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def model_fit(model=None, X_train_norm=None, Y_train=None, batch_size=1, epochs=1, device='gpu'):
    assert model is not None and X_train_norm is not None and Y_train is not None
    # TensorBoard
    callback_tb = keras.callbacks.TensorBoard(log_dir='dnn_log', histogram_freq=1, write_graph=True, write_images=True)
    # Келбек сохраняющий веса модели
    callback_as = keras.callbacks.ModelCheckpoint(filepath='last_training.okpt',
                                                  save_weights_only=True, verbose=1, period=5)
    # Келбек снижающий скорость обучения
    callback_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
    tf.device('/{}:0'.format(device))
    # заполнитель (обучение)
    model.fit(X_train_norm, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2,
                        callbacks=[callback_tb, callback_as, callback_lr])
    # сохранение модели
    model.save('last training model.h5')
    return model



def main():
    path = 'D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/обработанные данные/' \
           'КВАРТИРЫ Кронштадтский район.csv'
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
    X_test_norm =  scale_data(X_train, X_test)
    # Отрисовка данных
    show_data(X_train_norm)

    t = input('Режим\n1 | Обучение\n2 | Загрузка ыесов\n3 | загрузка модели')
    if t == '1':
        # создаем нейронную сеть
        model = create_model(X_train_norm.shape[1])
        # обучаем модель
        model = model_fit(model, X_train_norm, Y_train, 100, 200)
    elif t == '2':
        model = create_model(X_train_norm.shape[1])
        model.load_weights('last_training.okpt')
    else:
        model = keras.models.load_model('last training model.h5')

    model.summary()

    # оценка сети
    mse, mae = model.evaluate(X_test_norm, Y_test, verbose=0)
    print("Test mae: {}".format(mae))
    # предсказание
    pred = model.predict(X_test_norm)
    print(pred[1555][0], Y_test[1555])
    print(pred[1456][0], Y_test[1456])
    print(pred[1741][0], Y_test[1741])
    print(pred[354][0], Y_test[354])

if __name__ == '__main__':
    main()
