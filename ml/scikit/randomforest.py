from sklearn.ensemble import RandomForestClassifier
from ml.common.preprocessor import *
import joblib
import time


category = 'air-or-not'
version = 'fifth'
bandwidth = '20MHz'
object1 = 'air'
object2 = 'metal'
object3 = 'bottle'

accuracy_reduced = 0.0
accuracy_oob = 0.0
accuracy_raw = 0.0

opt_reduced = 0.0
opt_oob = 0.0

learn_reduced = 0.0
learn_oob = 0.0
learn_raw = 0.0

class_reduced = 0.0
class_oob = 0.0
class_raw = 0.0


for i in range(10):

    print('ATTEMPT #' + i)

    x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced, time_reduced = load_dataframe_reduced(category, version, bandwidth, object1, object2, object3)
    opt_reduced += time_reduced

    t1 = time.time()
    rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train_reduced, y_train_reduced)
    learn_reduced += (time.time() - t1)



    x_train_oob, y_train_oob, x_test_oob, y_test_oob, time_oob = load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2, object3)
    opt_oob += time_oob

    t2 = time.time()
    rfc_oob = RandomForestClassifier(max_depth=20).fit(x_train_oob, y_train_oob)
    learn_oob += (time.time() - t2)



    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataframe_raw(category, version, bandwidth, object1, object2, object3)

    t3 = time.time()
    rfc_raw = RandomForestClassifier(max_depth=20).fit(x_train_raw, y_train_raw)
    learn_raw += (time.time() - t3)


    t4 = time.time()
    score_reduced = rfc_reduced.score(x_test_reduced, y_test_reduced) * 100
    class_reduced += ((time.time() - t4) / 30200)
    accuracy_reduced += score_reduced

    t5 = time.time()
    score_oob = rfc_oob.score(x_test_oob, y_test_oob) * 100
    class_oob += ((time.time() - t5) / 30200)
    accuracy_oob += score_oob

    t6 = time.time()
    score_raw = rfc_raw.score(x_test_raw, y_test_raw) * 100
    class_raw += ((time.time() - t6) / 30200)
    accuracy_raw += score_raw

print('--------------------------------------------------------------------')
print('Accuracy reduced: ' + str(accuracy_reduced / 10))
print('Accuracy oob: ' + str(accuracy_oob / 10))
print('Accuracy raw: ' + str(accuracy_raw / 10))
print()
print('Opt time reduced: ' + str(opt_reduced / 10))
print('Opt time oob: ' + str(opt_oob / 10))
print()
print('Learn time reduced: ' + str(learn_reduced / 10))
print('Learn time oob: ' + str(learn_oob / 10))
print('Learn time raw: ' + str(learn_raw / 10))
print()
print('Classification time reduced: ' + str(class_reduced / 10))
print('Classification time oob: ' + str(class_oob / 10))
print('Classification time raw: ' + str(class_raw / 10))
print('--------------------------------------------------------------------')

# filename = '../model/rfc_reduced.joblib'
# joblib.dump(rfc_reduced, filename)
# filename = '../model/rfc_raw.joblib'
# joblib.dump(rfc_raw, filename)