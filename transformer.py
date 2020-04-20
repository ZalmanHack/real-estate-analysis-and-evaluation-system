import pickle
import os
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from datetime import datetime

class Transformer:
    def __init__(self, dir_root='DNN'):
        # корневой каталог
        self.dir_root = dir_root
        self.path = os.path.join(self.dir_root, 'Transformers')
        if not os.path.exists(self.dir_root):
            os.mkdir(self.dir_root)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        # инициализация трансформера (нормализатор).
        # Также его будет необходимо произвести заполнение тем данными, на которых была обучена модель
        self.transformer = QuantileTransformer(output_distribution='uniform')

    # загрузка (заполнение) трансформера. Принимает путь к бинарному файлу дампа
    def load_transformer(self, file_name=None):
        try:
            if file_name is None:
                file_name: str = datetime.today().strftime("%d.%m.%Y %H.%M")
            with open(os.path.join(self.path, '{}.bin'.format(file_name)), 'rb') as data:
                self.transformer = pickle.load(data)
        except Exception as e:
            print('load_transformer {}'.format(e))

    # сохранение трансформера в бинарный файл (создается дамп)
    def save_transformer(self, file_name=None):
        try:
            if file_name is None:
                file_name: str = datetime.today().strftime("%d.%m.%Y %H.%M")
            with open(os.path.join(self.path, '{}.bin'.format(file_name)), 'wb') as data:
                pickle.dump(self.transformer, data)
        except Exception as e:
            print('save_transformer {}'.format(e))

    def fit(self, X):
        self.transformer.fit(X)

    def transform(self, X):
        return self.transformer.transform(X)
