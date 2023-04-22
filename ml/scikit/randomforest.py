from IPython.core.display_functions import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from ml.common.preprocessor import *
import joblib
import time

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

category = 'air-or-not'
version = 'fifth'
bandwidth = '20MHz'
object1 = 'air'
object2 = 'metal'

start_time_reduced = time.time()
x_train_reduced, y_train_reduced, x_test_reduced, y_test_reduced = load_dataframe_reduced(category, version, bandwidth, object1, object2)
rfc_reduced = RandomForestClassifier(max_depth=20).fit(x_train_reduced, y_train_reduced)
print(x_train_reduced.head())
end_time_reduced = time.time() - start_time_reduced

for i in range(3):
    tree = rfc_reduced.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=x_train_reduced.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

start_time_oob = time.time()
x_train_oob, y_train_oob, x_test_oob, y_test_oob = load_dataframe_with_out_of_box_pca(category, version, bandwidth, object1, object2)
rfc_oob = RandomForestClassifier(max_depth=20).fit(x_train_oob, y_train_oob)
print(x_train_oob)
end_time_oob = time.time() - start_time_oob

start_time_raw = time.time()
x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_dataframe_raw(category, version, bandwidth, object1, object2)
print(x_train_raw.head())
rfc_raw = RandomForestClassifier(max_depth=20).fit(x_train_raw, y_train_raw)
end_time_raw = time.time() - start_time_raw

print('\n--------------------- REPORT ---------------------')
print("Accuracy reduced: " + str(rfc_reduced.score(x_test_reduced, y_test_reduced) * 100))
print("Reduced runtime: %s seconds" % (end_time_reduced))
print()
print("Accuracy with OOB PCA: " + str(rfc_oob.score(x_test_oob, y_test_oob) * 100))
print("OOB PCA runtime: %s seconds" % (end_time_oob))
print()
print("Accuracy raw: " + str(rfc_raw.score(x_test_raw, y_test_raw) * 100))
print("Raw runtime: %s seconds" % (end_time_raw))
print('--------------------------------------------------\n')

filename = '../model/rfc_reduced.joblib'
joblib.dump(rfc_reduced, filename)
filename = '../model/rfc_raw.joblib'
joblib.dump(rfc_raw, filename)