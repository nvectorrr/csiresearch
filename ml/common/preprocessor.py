from feature_selector import FeatureSelector
from ml.common.loader import load_train_and_test
from ml.common.out_of_box_pca import apply_pca
import time


def load_dataframe_reduced(cum_imtp, category, version, bandwidth, object1, object2, object3):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2, object3)
    labels = train['category']

    rot = time.time()
    fs = FeatureSelector(data=train.drop(['category', 'type'], axis=1), labels=labels)
    fs.identify_zero_importance(task='classification', eval_metric='multi_logloss',
                            n_iterations=10, early_stopping=True)

    fs.identify_low_importance(cumulative_importance=0.48)
    opt_time = time.time() - rot
    print('Reduced opt time: ' + str(opt_time))

    x_train = fs.remove(methods = {'zero_importance', 'low_importance'})
    removed = fs.check_removal()
    features_removed = x_train.shape[1]
    test_drop_filter = test.filter(removed)
    test.drop(test_drop_filter, inplace=True, axis=1)

    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test, opt_time, features_removed

def load_dataframe_with_pretrained_pca(category, version, bandwidth, object1, object2, object3):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2, object3)
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

def load_dataframe_raw(category, version, bandwidth, object1, object2, object3):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2, object3)

    x_train = train.drop(['type', 'category'], axis=1)
    y_train = train['category']
    x_test = test.drop(['type', 'category'], axis=1)
    y_test = test['category']

    return x_train, y_train, x_test, y_test

def load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2, object3, n_features):
    train, test = load_train_and_test(category, version, bandwidth, object1, object2, object3)

    oob = time.time()
    x_train, x_test = apply_pca(scaling=False, x_train=train.drop(['type', 'category'], axis=1), x_test=test.drop(['type', 'category'], axis=1), n_features=n_features)
    opt_time = time.time() - oob
    print('OOB opt time: ' + str(opt_time))
    y_train = train['category']
    y_test = test['category']

    return x_train, y_train, x_test, y_test, opt_time