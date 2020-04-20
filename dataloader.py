from sklearn.model_selection import train_test_split
from transformer import Transformer
import pandas as pd

class DataLoader:
    def __init__(self, root='DNN'):
        # инициализируем трасформер (нормлизатор) данных для нейронной сети
        self.tr = Transformer(root)

    def load(self, path, labels, test_size=0.2, random_state=42, fit_path=None):
        # загружаем датасет --------------------------------------------------------------------------------------------
        assert len(labels) > 0
        file = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        data = pd.DataFrame(file.loc[:, labels]).to_numpy()
        # разбиваем данные на 2 выборки: тренировочную и тестовую ------------------------------------------------------
        train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
        print('Размер датасета: {}\nРазмер тренировочной выборки: {}\nРазмер тестовой выборки: {}\n'.format(
            data.shape, train_set.shape, test_set.shape))
        # разделяем на X и Y обе выборки. Где X - входы. Y - выходы
        X_train, Y_train = train_set[:, :2], train_set[:, 2].reshape(-1, 1)
        X_test, Y_test = test_set[:, :2], test_set[:, 2].reshape(-1, 1)
        print('Тренировочная выборка:\n Входы: {}\n Выходы: {}\n'.format(X_train.shape, Y_train.shape))
        print('Тестовая выборка:\n Входы: {}\n Выходы: {}\n'.format(X_test.shape, Y_test.shape))
        # нормализация данных ------------------------------------------------------------------------------------------
        if fit_path is None:  # если сохраненного файла заполнителя нет, то заполняем прочитанными данными
            self.tr.fit(X_train)  # заполняем его
        else:
            self.tr.load_transformer(fit_path)
        X_train_norm = self.tr.transform(X_train)  # нормализируем тренеровочную выборку
        X_test_norm = self.tr.transform(X_test)  # нормализируем тестовую выборку
        return (X_train_norm, Y_train), (X_test_norm, Y_test)