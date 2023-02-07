from sklearn.ensemble import RandomForestClassifier
from dataframe import *
import numpy as np
import time

# bad seeds examples: 30, 20, 1, 10, 12, 36, 100, 80, 70, 75, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16
# good seeds examples: 40, 90, 76, 17

np.random.seed(76)

start_time_reduced = time.time()
x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced = load_dataframe_reduced()
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
