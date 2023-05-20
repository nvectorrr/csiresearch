from sklearn.svm import SVC
from ml.common.preprocessor import *
from sklearn.metrics import accuracy_score
import joblib
import time


category = 'air-or-not'
version = 'fifth'
bandwidth = '20MHz'
object1 = 'air'
object2 = 'metal'

start_time_reduced = time.time()
x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced = load_dataframe_reduced(category, version, bandwidth, object1, object2)
svm_reduced = SVC(kernel = 'linear').fit(x_train_reduced, y_train_reduced)
end_time_reduced = time.time() - start_time_reduced

start_time_oob = time.time()
x_train_oob, y_train_oob, x_test_oob, y_test_oob = load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2)
svm_oob = SVC(kernel = 'linear').fit(x_train_oob, y_train_oob)
end_time_oob = time.time() - start_time_oob

start_time_combined = time.time()
x_train_combined, y_train_combined, x_test_combined, y_test_combined = load_dataframe_combined(category, version, bandwidth, object1, object2)
svm_combined = SVC(kernel = 'linear').fit(x_train_combined, y_train_combined)
end_time_combined = time.time() - start_time_combined

start_time_raw = time.time()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataframe_raw(category, version, bandwidth, object1, object2)
svm_raw = SVC(kernel = 'linear').fit(x_train_raw, y_train_raw)
end_time_raw = time.time() - start_time_raw

reduced_pred = svm_reduced.predict(x_test_reduced)
oob_pred = svm_oob.predict(x_test_oob)
combined_pred = svm_combined.predict(x_test_combined)
raw_pred = svm_raw.predict(x_test_raw)

print('\n--------------------- REPORT ---------------------')
print("Accuracy reduced: " + str(accuracy_score(y_test_reduced, reduced_pred) * 100))
print("Reduced runtime: %s seconds" % (end_time_reduced))
print()
print("Accuracy with OOB PCA: " + str(accuracy_score(y_test_oob, oob_pred) * 100))
print("OOB PCA runtime: %s seconds" % (end_time_oob))
print()
print("Accuracy with combined PCA: " + str(accuracy_score(y_test_combined, combined_pred) * 100))
print("Combined PCA runtime: %s seconds" % (end_time_combined))
print()
print("Accuracy raw: " + str(accuracy_score(y_test_raw, raw_pred) * 100))
print("Raw runtime: %s seconds" % (end_time_raw))
print('--------------------------------------------------\n')

filename = '../model/svm_reduced.joblib'
joblib.dump(svm_reduced, filename)
filename = '../model/svm_oob.joblib'
joblib.dump(svm_oob, filename)
filename = '../model/svm_combined.joblib'
joblib.dump(svm_combined, filename)
filename = '../model/svm_raw.joblib'
joblib.dump(svm_raw, filename)