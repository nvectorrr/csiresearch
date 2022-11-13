import processing.process
import pandas as pd


def load_dataframe():
    train = load_csi('../../datasets/air-or-not/air_train_2.dat', 'train', 'air')
    test = load_csi('../../datasets/air-or-not/pc_case_test.dat', 'test', 'case')
    test2 = load_csi('../../datasets/air-or-not/air_test.dat', 'test', 'air')

    test = pd.concat([test, test2], axis=0)

    print(test.head())
    print(test.tail())

    test = test.sample(frac=1).reset_index(drop=True)

    x_train = train.drop(['type', 'category'], axis=1)
    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test


def load_csi(path, type, category):
    dataframe = get_dataframe(path)
    dataframe['type'] = type
    dataframe['category'] = category

    return dataframe


def get_dataframe(path):
    [csi, data] = processing.read.extractCSI(path)  # path to test file
    csi = processing.process.extractAm(csi)
    csi = processing.process.reshape224x1(csi)

    dataframe = pd.DataFrame(csi)

    return dataframe