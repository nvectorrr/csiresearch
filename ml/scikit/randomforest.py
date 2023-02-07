from sklearn.ensemble import RandomForestClassifier
from dataframe import *
import time

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