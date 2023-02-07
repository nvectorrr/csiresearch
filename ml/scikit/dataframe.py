import processing.process
from feature_selector import FeatureSelector
import pandas as pd


def load_dataframe_reduced():
    #train = load_csi('../../datasets/air-or-not/fourth/40MHz/metal.dat', 'train', 'metal')
    #train2 = load_csi('../../datasets/air-or-not/fourth/40MHz/air.dat', 'train', 'air')
    #test = load_csi('../../datasets/air-or-not/fourth/40MHz/metal_test.dat', 'test', 'metal')
    #test2 = load_csi('../../datasets/air-or-not/fourth/40MHz/air_test.dat', 'test', 'air')

    train = load_csi('../../datasets/air-or-not/fifth/20MHz/metal_train.dat', 'train', 'metal')
    train2 = load_csi('../../datasets/air-or-not/fifth/20MHz/air_train.dat', 'train', 'air')
    test = load_csi('../../datasets/air-or-not/fifth/20MHz/metal_test.dat', 'test', 'metal')
    test2 = load_csi('../../datasets/air-or-not/fifth/20MHz/air_test.dat', 'test', 'air')

    train = pd.concat([train, train2], axis=0)
    test = pd.concat([test, test2], axis=0)

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    labels = train['category']

    fs = FeatureSelector(data=train, labels=labels)
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                            n_iterations=10, early_stopping=True)

    fs.identify_low_importance(cumulative_importance=0.95)

    train = fs.remove(methods = {'zero_importance', 'low_importance'})
    removed = fs.check_removal()
    test_drop_filter = test.filter(removed)
    test.drop(test_drop_filter, inplace=True, axis=1)

    x_train = train.drop(['type', 'category', 'category_air'], axis=1)
    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test

def load_dataframe_raw():
    train = load_csi('../../datasets/air-or-not/fifth/20MHz/metal_train.dat', 'train', 'metal')
    train2 = load_csi('../../datasets/air-or-not/fifth/20MHz/air_train.dat', 'train', 'air')
    test = load_csi('../../datasets/air-or-not/fifth/20MHz/metal_test.dat', 'test', 'metal')
    test2 = load_csi('../../datasets/air-or-not/fifth/20MHz/air_test.dat', 'test', 'air')

    train = pd.concat([train, train2], axis=0)
    test = pd.concat([test, test2], axis=0)

    train = train.sample(frac=1).reset_index(drop=True)
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