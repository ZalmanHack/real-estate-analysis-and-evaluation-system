import pickle
import os
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from datetime import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Type:
    quantileTransformer: str = 'QuantileTransformer'
    oneHotEncoder: str = 'OneHotEncoder'
    ordinalEncoder: str = 'OrdinalEncoder'
    minMaxScaler: str = 'MinMaxScaler'


class Transformer:
    def __init__(self, dir_root='DNN DATA', transformer='QuantileTransformer'):
        self.file_name = transformer
        self.transformers = {
            'QuantileTransformer': QuantileTransformer(output_distribution='uniform'),  # трансформер для координат
            'OneHotEncoder': OneHotEncoder(sparse=False),  # трансформер для категорий
            'OrdinalEncoder': OrdinalEncoder(),
            'MinMaxScaler': MinMaxScaler()}
        assert transformer in self.transformers
        # инициализация трансформера (нормализатор).
        # Также его будет необходимо произвести заполнение тем данными, на которых была обучена модель
        # трансформер для координат
        self.transformer = self.transformers[transformer]
        # корневой каталог
        self.dir_root = dir_root
        self.path = os.path.join(self.dir_root, 'Transformers')
        if not os.path.exists(self.dir_root):
            os.mkdir(self.dir_root)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    # загрузка (заполнение) трансформера. Принимает путь к бинарному файлу дампа
    def load_transformer(self, file_name=None):
        try:
            if file_name is None:
                file_name: str = self.file_name
            with open(os.path.join(self.path, '{}.bin'.format(file_name)), 'rb') as data:
                self.transformer = pickle.load(data)
                return self.transformer
        except Exception as e:
            print('load_transformer {}'.format(e))

    # сохранение трансформера в бинарный файл (создается дамп)
    def save_transformer(self, file_name=None):
        try:
            if file_name is None:
                file_name: str = self.file_name
            with open(os.path.join(self.path, '{}.bin'.format(file_name)), 'wb') as data:
                pickle.dump(self.transformer, data)
        except Exception as e:
            print('save_transformer {}'.format(e))

    def fit(self, X):
        self.transformer.fit(X)

    def transform(self, X):
        return self.transformer.transform(X)

    def inverse_transform(self, X):
        return self.transformer.inverse_transform(X)
