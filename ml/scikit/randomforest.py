from sklearn.ensemble import RandomForestClassifier
from dataframe import *

x_train, y_train, x_test, y_test = load_dataframe()

print(x_train)

rfc = RandomForestClassifier(max_depth=20).fit(x_train, y_train)
print(rfc.score(x_test, y_test) * 100)
