
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from mpldatacursor import datacursor
from dataloader import DataLoader
from transformer import Transformer

dataLoader = DataLoader()
path = 'D:/Documents/учёба/КНТ/4 курс 2 семестр/dataset недвижимость СПБ/обработанные данные/РАЙОНЫ2.csv'
fit_path = None
# загружаем датасет
(X_train, Y_train), (X_test, Y_test) = dataLoader.load(path=path, labels=['Широта', 'Долгота', 'Район'], test_size=0.2, fit_path=fit_path)

# нормализуем координаты тем же методом, что и для нейронной сети
x_transformer = Transformer(transformer='QuantileTransformer')
x_transformer.fit(X_train)
X_train_norm = x_transformer.transform(X_train)

# нормализуем классы в вектора [ 2. 16.  7. ...  9. 16.  7.]
y_transformer = Transformer(transformer='OrdinalEncoder')
y_transformer.fit(Y_train)
y_train_norm = y_transformer.transform(Y_train).reshape(-1)

labels = y_transformer.transformer.categories_[0]

print(labels)

# Определяем количество соседей для оценки
n_neighbors = 6
# Определяем шаг сетки
h = .001  # step size in the mesh

for weights in ['uniform', 'distance']:
    # Создание классификатора K-ближайших и заполняем его
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X_train_norm, y_train_norm)

    # График решения границы. Для этого мы назначим цвет каждому
    # точка в сетке [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    # Генерируем сетку с шагом h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # запоминаем размерность матрицы для дальнейшего отображения
    shape = xx.shape

    # получаем классификацию
    # np.c_[] - склеивает столбцы в одну матрицу
    # np.ravel() - преобразовывает массив в одномерный
    vector = x_transformer.transform(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict(vector)

    # Преобразовываем результат в матрицу
    Z = Z.reshape(shape)
    plt.figure()
    plt.pcolormesh(yy, xx, Z)  # , cmap=cmap_light)



    # Plot also the training points
    # plt.scatter(X_train[:, 1], X_train[:, 0], c=y_train_norm, # cmap=cmap_bold,
    #             edgecolor='k', s=20)

    plt.xlim(yy.min(), yy.max())
    plt.ylim(xx.min(), xx.max())
    plt.title("{}-Class classification (k = {}, weights = '{}')".format(len(labels),
                                                                        n_neighbors, weights))

    datacursor(hover=False, bbox=dict(alpha=1, fc='w'), formatter='{z:.06g}'.format)
plt.show()
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import csv

data_names = ['Москва', 'Санкт-Петербург', 'Сочи', 'Архангельск',
              'Владимир', 'Краснодар', 'Курск', 'Воронеж',
              'Ставрополь', 'Мурманск']
data_values = [1076, 979, 222, 189, 137, 134, 124, 124, 91, 79]

dpi = 80
fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
mpl.rcParams.update({'font.size': 9})

plt.title('Распределение кафе по городам России (%)')

xs = range(len(data_names))

plt.pie(
    data_values, autopct='%.1f', radius = 1.1,
    explode = [0.15] + [0 for _ in range(len(data_names) - 1)] )
plt.legend(
    bbox_to_anchor = (-0.16, 0.45, 0.25, 0.25),
    loc = 'lower left', labels = data_names )
plt.show()
'''