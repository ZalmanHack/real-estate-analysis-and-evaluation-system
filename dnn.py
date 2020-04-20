import os
import sys
import pandas as pd
from dataclasses import dataclass
from tensorflow import keras, train
from datetime import datetime

# класс DNN (Deep Neural Network) - позволяет работать с различными моделями нейронных сетей.
# Основан на библиотеке Tensorflow версии 2.0.0 с API Keras.
# Вычисления по возможности выполняются на GPU.
class Dnn:
    def __init__(self, dir_root='DNN', model=None, transformer=None):
        # инициализация папок
        self.dir_root = dir_root
        self.dir_weights = os.path.join(self.dir_root, 'Weights')
        self.dir_models = os.path.join(self.dir_root, 'Models')
        self.dir_logs = os.path.join(self.dir_root, 'Logs')
        self.dir_last_train = None  # папка последнего обучения (заполнения) сети
        if not os.path.exists(self.dir_root):
            os.mkdir(self.dir_root)
        if not os.path.exists(self.dir_weights):
            os.mkdir(self.dir_weights)
        if not os.path.exists(self.dir_models):
            os.mkdir(self.dir_models)
        if not os.path.exists(self.dir_logs):
            os.mkdir(self.dir_logs)
        self.model = None

    # компоновка и компиляция нейронной сети
    def create(self, inputs=2, outputs=1, layers: list = (256, 128), activations: list = ('relu', 'relu')):
        assert inputs > 0
        assert outputs > 0
        assert len(activations) > 1, len(activations) == len(layers)
        self.model = keras.Sequential()
        # первый слой --------------------------------------------------------------------------------------------------
        self.model.add(keras.layers.Dense(layers[0], input_dim=inputs, kernel_initializer='glorot_uniform'))
        self.model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
        self.model.add(keras.layers.Activation(activations[0]))
        for i in range(1, len(layers)):
            # первый слой ----------------------------------------------------------------------------------------------
            self.model.add(keras.layers.Dense(layers[i], use_bias=True, kernel_initializer='glorot_uniform'))
            self.model.add(keras.layers.BatchNormalization(epsilon=1e-6, center=True, scale=True))
            self.model.add(keras.layers.Activation(activations[i]))
            # self.model.add(keras.layers.Dropout(0.2))
        # четвертый слой -----------------------------------------------------------------------------------------------
        self.model.add(keras.layers.Dense(outputs))
        # компилируем нейронную сеть
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        return self.model

    # подключение калбеков и обучение модели
    def fit(self, X_train_norm=None, Y_train=None, batch_size=1, epochs=1, dir_name=None):
        assert self.model is not None and X_train_norm is not None and Y_train is not None
        assert X_train_norm.shape[0] == Y_train.shape[0]
        # TensorBoard --------------------------------------------------------------------------------------------------
        callback_tb = keras.callbacks.TensorBoard(log_dir=self.dir_logs, histogram_freq=1,
                                                  write_graph=True, write_images=True)
        # Келбек сохраняющий веса модели -------------------------------------------------------------------------------
        # Добавим эпоху в имя файла (uses `str.format`)
        if dir_name is None:
            dir_name: str = datetime.today().strftime("%d.%m.%Y %H.%M")
        path = os.path.join(self.dir_weights, dir_name, "cp-{epoch:04d}.ckpt")
        callback_as = keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True,
                                                      verbose=1, period=2)
        # Келбек снижающий скорость обучения ---------------------------------------------------------------------------
        callback_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                        patience=2, min_lr=0.0005)
        #tf.device('/{}:0'.format(device))
        # Сохраним веса 1 эпохи, используя формат `checkpoint_path` format ---------------------------------------------
        self.model.save_weights(path.format(epoch=0))
        # запоминаем путь к папке последнего сохранения ----------------------------------------------------------------
        self.dir_last_train = os.path.join(self.dir_weights, dir_name)
        # заполнитель (обучение) ---------------------------------------------------------------------------------------
        self.model.fit(X_train_norm, Y_train, batch_size=batch_size,
                       epochs=epochs, verbose=2, validation_split=0.1,
                       callbacks=[callback_tb, callback_as, callback_lr])
        # сохранение модели
        self.model.save(os.path.join(self.dir_models, '{date}.h5'.format(date=dir_name)))
        return self.model

    # оценка точнсти сети
    def evaluate(self, X_norm=None, Y=None):
        assert self.model is not None and X_norm is not None and Y is not None
        mse, mae = self.model.evaluate(X_norm, Y, verbose=0)
        return mse, mae

    # Применение сети
    def predict(self, X_norm):
        assert self.model is not None and X_norm is not None
        return self.model.predict(X_norm)

    def _get_last_saved_dir(self, path):
        try:
            list_dir = pd.DataFrame(os.listdir(path))
            list_dir = list_dir.sort_values(by=[0])
            list_dir = list_dir.reset_index(drop=True)
            return os.path.join(path, str(list_dir.iloc[-1, 0]))
        except Exception as e:
            print('_get_last_saved_dir: {}'.format(e))


    def load_weights(self, path=None):
        try:
            if path is None:
                path = self._get_last_saved_dir(self.dir_weights)
            lastest = train.latest_checkpoint(path)  # weights 04.18.2020 21.45
            # lastest = train.latest_checkpoint('best weights')
            assert lastest is not None, 'При загрузке весов произошла ошибка: lastest is None'
            self.model.load_weights(lastest)
        except Exception as e:
            print(e)
            sys.exit(-1)

    # путь к последней папке тренировки
    def get_dir_last_train(self):
        return self.dir_last_train

    # сохранение модели
    def save_model(self, dir_name=None):
        assert dir_name is not None
        self.model.save(os.path.join(self.dir_models, '{date}.h5'.format(date=dir_name)))

    def load_model(self, dir_name=None):
        try:
            if dir_name is None:
                dir_name = self._get_last_saved_dir(self.dir_models)
            assert dir_name is not None, 'При загрузке весов произошла ошибка: dir_name is None'
            self.model = keras.models.load_model(dir_name)
        except Exception as e:
            print('load_model: {}'.format(e))
            sys.exit(-1)
