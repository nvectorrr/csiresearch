from sklearn.neighbors import KNeighborsClassifier
from ml.common.preprocessor import *
import joblib
import time


category = 'air-or-not'
version = 'fifth'
bandwidth = '20MHz'
object1 = 'air'
object2 = 'metal'

start_time_reduced = time.time()
x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced = load_dataframe_reduced(category, version, bandwidth, object1, object2)
knn_reduced = KNeighborsClassifier(n_neighbors=5).fit(x_train_reduced, y_train_reduced)
end_time_reduced = time.time() - start_time_reduced

start_time_oob = time.time()
x_train_oob, y_train_oob, x_test_oob, y_test_oob = load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2)
knn_oob = KNeighborsClassifier(n_neighbors=5).fit(x_train_oob, y_train_oob)
end_time_oob = time.time() - start_time_oob

start_time_combined = time.time()
x_train_combined, y_train_combined, x_test_combined, y_test_combined = load_dataframe_combined(category, version, bandwidth, object1, object2)
knn_combined = KNeighborsClassifier(n_neighbors=5).fit(x_train_combined, y_train_combined)
end_time_combined = time.time() - start_time_combined

start_time_raw = time.time()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataframe_raw(category, version, bandwidth, object1, object2)
knn_raw = KNeighborsClassifier(n_neighbors=5).fit(x_train_raw, y_train_raw)
end_time_raw = time.time() - start_time_raw

print('\n--------------------- REPORT ---------------------')
print("Accuracy reduced: " + str(knn_reduced.score(x_test_reduced, y_test_reduced) * 100))
print("Reduced runtime: %s seconds" % (end_time_reduced))
print()
print("Accuracy with OOB PCA: " + str(knn_oob.score(x_test_oob, y_test_oob) * 100))
print("OOB PCA runtime: %s seconds" % (end_time_oob))
print()
print("Accuracy with combined PCA: " + str(knn_combined.score(x_test_combined, y_test_combined) * 100))
print("Combined PCA runtime: %s seconds" % (end_time_combined))
print()
print("Accuracy raw: " + str(knn_raw.score(x_test_raw, y_test_raw) * 100))
print("Raw runtime: %s seconds" % (end_time_raw))
print('--------------------------------------------------\n')

filename = '../model/knn_reduced.joblib'
joblib.dump(knn_reduced, filename)
filename = '../model/knn_oob.joblib'
joblib.dump(knn_oob, filename)
filename = '../model/knn_combined.joblib'
joblib.dump(knn_combined, filename)
filename = '../model/knn_raw.joblib'
joblib.dump(knn_raw, filename)