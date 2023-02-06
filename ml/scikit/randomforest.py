from sklearn.ensemble import RandomForestClassifier
from dataframe import *

x_train, y_train, x_test, y_test = load_dataframe_reduced()

rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train, y_train)
print("Accuracy reduced: " + str(rfc_reduced.score(x_test, y_test) * 100))

x_train, y_train, x_test, y_test = load_dataframe_raw()

rfc_raw = RandomForestClassifier(max_depth=20).fit(x_train, y_train)
print("Accuracy raw: " + str(rfc_raw.score(x_test, y_test) * 100))