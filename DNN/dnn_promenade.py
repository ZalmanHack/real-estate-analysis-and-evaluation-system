from Dnn import Dnn
from transformer import Transformer, Type


class DnnPromenade(Dnn):
    def __init__(self, dir_root='DNN DATA'):
        super().__init__(dir_root)
        self.model_name = 'distance to promenade'
        self.inputs = 2
        self.outputs = 1
        self.layers = [256, 128, 128, 128, 128, 128]
        self.activations = ['relu', 'relu', 'elu', 'elu', 'elu', 'elu']
        self.batch_size = 40
        # Нормализация
        self.x_tr = None

    def create(self):
        return super().create_regression(self.inputs, self.outputs, self.layers, self.activations)

    # обучение нейронной сети
    def fit(self, X=None, Y=None, batch_size=40, epochs=100, dir_name='distance to promenade'):
        if self.x_tr is None:                                                 # Если нормализатора еще нет
            self.x_tr = Transformer(self.dir_root, Type.quantileTransformer)  # Создаем его
            self.x_tr.fit(X)                                                  # Заполняем нормализатор
        X_norm = self.x_tr.transform(X)                                       # Нормализуем X
        return super().fit(X_norm, Y, batch_size, epochs, dir_name)         # Обучаем НС и возвращаем готовую модель

    def evaluate(self, X=None, Y=None):
        X_norm = self.x_tr.transform(X)
        return super().evaluate(X_norm, Y)

    def predict(self, X):
        X_norm = self.x_tr.transform(X)
        return super().predict(X_norm)

    def load_weights(self, path='distance to promenade'):
        super().load_weights(path)

    def load_model(self, path='distance to promenade'):
        super().load_model(path)

    def save_model(self, path='distance to promenade'):
        super().save_model(path)

    def load_x_transformer(self, tr=None, tr_path='coordinates'):
        assert (tr is not None and tr_path is None) or (tr is None and tr_path is not None)
        if tr_path:
            self.x_tr = Transformer(self.dir_root, Type.quantileTransformer)
            self.x_tr.load_transformer(file_name=tr_path)
        elif tr:
            self.x_tr = tr

    def save_x_transformer(self, tr_path='coordinates'):
        assert tr_path is not None
        self.x_tr.save_transformer(file_name=tr_path)