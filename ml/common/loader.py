import processing.process
import pandas as pd


available_categories = ['air-or-not', 'first']
available_versions = ['first', 'second', 'third', 'fourth', 'fifth']
available_bandwidth = ['20MHz', '40MHz']
available_objects = ['air', 'case', 'bottle', 'metal', 'case']

def load_train_and_test(category, version, bandwidth, object1, object2):
    base_path_1 = '../../datasets/'

    if category in available_categories:
        base_path_1 += category
        base_path_1 += '/'
    else:
        raise FileNotFoundError('Unknown category!')

    if version in available_versions:
        base_path_1 += version
        base_path_1 += '/'
    else:
        raise FileNotFoundError('Unknown version!')

    if bandwidth in available_bandwidth:
        base_path_1 += bandwidth
        base_path_1 += '/'
    else:
        raise FileNotFoundError('Unknown bandwidth!')

    base_path_2 = base_path_1

    if object1 in available_objects:
        base_path_1 += object1
    else:
        raise FileNotFoundError('Unknown object!')

    if object2 in available_objects:
        base_path_2 += object2
    else:
        raise FileNotFoundError('Unknown object!')

    train1 = load_csi(base_path_1 + '_train.dat', 'train', object1)
    train2 = load_csi(base_path_2 + '_train.dat', 'train', object2)
    test1 = load_csi(base_path_1 + '_test.dat', 'test', object1)
    test2 = load_csi(base_path_2 + '_test.dat', 'test', object2)

    train = pd.concat([train1, train2], axis=0)
    test = pd.concat([test1, test2], axis=0)

    print(train.head())

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, test

def load_csi(path, type, category):
    dataframe = get_dataframe(path)
    dataframe['type'] = type
    dataframe['category'] = category

    return dataframe

def get_dataframe(path):
    [csi, data] = processing.read.extractCSI(path)
    csi = processing.process.extractAm(csi)
    csi = processing.process.reshape224x1(csi)

    dataframe = pd.DataFrame(csi)

    return dataframe