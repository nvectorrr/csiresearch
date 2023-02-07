from sklearn.ensemble import RandomForestClassifier
from dataframe import *
import numpy as np
import time

# bad seeds examples: 30, 20, 1, 10, 12, 36, 100, 80, 70, 75, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16
# good seeds examples: 40, 90, 76, 17

np.random.seed(90)

start_time = time.time()
x_train, y_train, x_test, y_test = load_dataframe_reduced()
rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train, y_train)
print("Accuracy reduced: " + str(rfc_reduced.score(x_test, y_test) * 100))
print("Reduced runtime: %s seconds" % (time.time() - start_time))

start_time = time.time()
x_train, y_train, x_test, y_test = load_dataframe_raw()
rfc_raw = RandomForestClassifier(max_depth=20).fit(x_train, y_train)
print("Accuracy raw: " + str(rfc_raw.score(x_test, y_test) * 100))
print("Raw runtime: %s seconds" % (time.time() - start_time))
