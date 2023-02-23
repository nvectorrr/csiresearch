from sklearn.ensemble import RandomForestClassifier
from ml.common.preprocessor import *
import joblib
import time


category = 'air-or-not'
version = 'fifth'
bandwidth = '40MHz'
object1 = 'air'
object2 = 'metal'
object3 = 'bottle'

accuracy_reduced = 0.0
accuracy_oob = 0.0

opt_reduced = 0.0
opt_oob = 0.0

learn_reduced = 0.0
learn_oob = 0.0

class_reduced = 0.0
class_oob = 0.0

list_accuracy_reduced = list()
list_accuracy_obb = list()
#list_accuracy_reduced_raw = list()

n_features_reduced_list = list()
n_features_oob_list = list()
#n_features_raw = list()


for i in range(0.1, 0.9, 0.05):

    print('ATTEMPT #' + str(i))

    x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, time_reduced, features_removed = load_dataframe_reduced(i, category, version, bandwidth, object1, object2, object3)
    opt_reduced += time_reduced

    t1 = time.time()
    rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train_reduced, y_train_reduced)
    learn_reduced += (time.time() - t1)



    x_train_oob, y_train_oob, x_test_oob, y_test_oob, time_oob = load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2, object3, n_features=features_removed)
    opt_oob += time_oob

    t2 = time.time()
    rfc_oob = RandomForestClassifier(max_depth=20).fit(x_train_oob, y_train_oob)
    learn_oob += (time.time() - t2)


    t4 = time.time()
    score_reduced = rfc_reduced.score(x_test_reduced, y_test_reduced) * 100
    class_reduced += ((time.time() - t4) / 30200)
    list_accuracy_reduced.append(score_reduced)
    accuracy_reduced += score_reduced

    t5 = time.time()
    score_oob = rfc_oob.score(x_test_oob, y_test_oob) * 100
    class_oob += ((time.time() - t5) / 30200)
    list_accuracy_obb.append(score_oob)
    accuracy_oob += score_oob

    n_features_reduced_list.append(features_removed)
    n_features_oob_list.append(features_removed)

print('--------------------------------------------------------------------')
print('Accuracy reduced: ' + str(accuracy_reduced))
print('Accuracy oob: ' + str(accuracy_oob))
print()
print('Opt time reduced: ' + str(opt_reduced))
print('Opt time oob: ' + str(opt_oob))
print()
print('Learn time reduced: ' + str(learn_reduced))
print('Learn time oob: ' + str(learn_oob))
print()
print('Classification time reduced: ' + str(class_reduced))
print('Classification time oob: ' + str(class_oob))
print('--------------------------------------------------------------------')

# filename = '../model/rfc_reduced.joblib'
# joblib.dump(rfc_reduced, filename)
# filename = '../model/rfc_raw.joblib'
# joblib.dump(rfc_raw, filename)