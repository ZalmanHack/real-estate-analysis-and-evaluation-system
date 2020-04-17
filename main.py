import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    data = pd.read_csv("D:\\Documents\\учёба\\КНТ\\4 курс 2 семестр\\"
                       "dataset недвижимость СПБ\\обработанные данные\\"
                       "КВАРТИРЫ Калининский район.csv", sep=';', encoding='utf-8-sig')
    print(data.info())
    reshape_data = data.to_numpy()[:, 6:8]
    print(reshape_data)

    scaler = preprocessing.MinMaxScaler()
    n_data = scaler.fit_transform(reshape_data)
    print(n_data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.scatter(n_data[:, 0], n_data[:, 1], c='r', s=3)
    print(n_data[:100, 1], n_data[:100, 0])

    plt.show()
    pass


if __name__ == "__main__":
    main()
