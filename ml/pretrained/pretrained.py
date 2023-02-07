from sklearn.ensemble import RandomForestClassifier
from ml.common.dataframe import *
import time

start_time_reduced = time.time()
x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced = load_dataframe_with_pretrained_pca()
rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train_reduced, y_train_reduced)
end_time_reduced = time.time() - start_time_reduced

start_time_raw = time.time()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataframe_raw()
rfc_raw = RandomForestClassifier(max_depth=20).fit(x_train_raw, y_train_raw)
end_time_raw = time.time() - start_time_raw

print('\n------------- REPORT -------------')
print("Accuracy reduced: " + str(rfc_reduced.score(x_test_reduced, y_test_reduced) * 100))
print("Reduced runtime: %s seconds" % (end_time_reduced))
print()
print("Accuracy raw: " + str(rfc_raw.score(x_test_raw, y_test_raw) * 100))
print("Raw runtime: %s seconds" % (end_time_raw))
print('----------------------------------\n')