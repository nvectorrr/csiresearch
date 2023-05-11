import pandas as pd

from feature_selector import FeatureSelector
from ml.common.loader import load_train_and_test
from ml.common.out_of_box_pca import apply_pca, apply_pca_for_some_comp


def load_dataframe_reduced(category, version, bandwidth, object1, object2):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2)
    labels = train['category']

    fs = FeatureSelector(data=train.drop(['category', 'type'], axis=1), labels=labels)
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                            n_iterations=10, early_stopping=True)

    fs.identify_low_importance(cumulative_importance=0.70)

    x_train = fs.remove(methods = {'zero_importance', 'low_importance'})
    removed = fs.check_removal()
    test_drop_filter = test.filter(removed)
    test.drop(test_drop_filter, inplace=True, axis=1)

    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test

def load_dataframe_with_pretrained_pca(category, version, bandwidth, object1, object2):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2)
    labels = train['category']

    fs = FeatureSelector(data=train.drop(['category', 'type'], axis=1), labels=labels)
    fs.select_feature_pretrained(n_iterations=10)

    fs.identify_low_importance(cumulative_importance=0.70)

    x_train = fs.remove(methods={'zero_importance', 'low_importance'})
    removed = fs.check_removal()
    test_drop_filter = test.filter(removed)
    test.drop(test_drop_filter, inplace=True, axis=1)

    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test

def load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2)

    x_train, x_test = apply_pca(scaling=False, x_train=train.drop(['type', 'category'], axis=1), x_test=test.drop(['type', 'category'], axis=1))
    y_train = train['category']
    y_test = test['category']

    return x_train, y_train, x_test, y_test

def load_dataframe_combined(category, version, bandwidth, object1, object2):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2)

    if bandwidth == '20MHz':
        num_comp = 112
    else:
        num_comp = 228

    print(train.head())

    x_train, x_test = apply_pca_for_some_comp(scaling=False, x_train=train.drop(['type', 'category'], axis=1),
                                x_test=test.drop(['type', 'category'], axis=1), n_comp=num_comp)
    y_train = train['category']
    y_test = test['category']

    new_train = pd.DataFrame(x_train)
    new_test = pd.DataFrame(x_test)
    new_train['category'] = y_train
    new_test['category'] = y_test

    labels = new_train['category']

    fs = FeatureSelector(data=new_train.drop(['category'], axis=1), labels=labels)
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                                n_iterations=10, early_stopping=True)

    fs.identify_low_importance(cumulative_importance=0.75)

    new_x_train = fs.remove(methods={'zero_importance', 'low_importance'})
    removed = fs.check_removal()
    test_drop_filter = new_test.filter(removed)
    new_test.drop(test_drop_filter, inplace=True, axis=1)

    new_y_train = new_train['category']
    new_x_test = new_test.drop(['category'], axis=1)
    new_y_test = new_test['category']

    return new_x_train, new_y_train, new_x_test, new_y_test

def load_dataframe_raw(category, version, bandwidth, object1, object2):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2)

    x_train = train.drop(['type', 'category'], axis=1)
    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test