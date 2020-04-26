from Dnn import Dnn

class DnnRegions(Dnn):
    def __init__(self, dir_root='DNN DATA'):
        super().__init__(dir_root)
        self.model_name = 'distance to center'
        self.inputs = 2
        self.outputs = 18
        self.layers = [128, 128, 64]
        self.activations = ['elu', 'elu', 'sigmoid']
        self.batch_size = 100

    def create(self):
        return super().create_categorical(self.inputs, self.outputs, self.layers, self.activations)

    def fit(self, X_train_norm=None, Y_train=None, batch_size=50, epochs=100, dir_name='regions'):
        return super().fit(X_train_norm, Y_train, batch_size, epochs, dir_name)

    def load_weights(self, path='regions'):
        super().load_weights(path)

    def load_model(self, path='regions.h5'):
        super().load_model(path)

    def save_model(self, path='regions.h5'):
        super().save_model(path)
