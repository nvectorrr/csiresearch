import processing.process
import pandas as pd


def load_dataframe():
    train = load_train()
    test = load_test()

    x_train = train.drop(['type', 'category'], axis=1)
    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test


def load_test():
    [csi, data] = processing.read.extractCSI('../../datasets/data1.dat')  # path to test file
    csi = processing.process.extractAm(csi)
    csi = processing.process.reshape224x1(csi)

    dataframe = pd.DataFrame(csi)
    dataframe['type'] = 'test'
    dataframe['category'] = 'air'

    return dataframe


def load_train():
    [csi, data] = processing.read.extractCSI('../../datasets/data2.dat')  # path to training file
    csi = processing.process.extractAm(csi)
    csi = processing.process.reshape224x1(csi)

    dataframe = pd.DataFrame(csi)
    dataframe['type'] = 'train'
    dataframe['category'] = 'air'

    return dataframe

# TODO get some more datasets (with(out) bottle)