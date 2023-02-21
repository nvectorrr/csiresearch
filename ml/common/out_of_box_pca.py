from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_pca(scaling, x_train, x_test):
    if scaling == True:
        x_train, x_test = scale_data(x_train, x_test)

    pca = PCA(n_components=30)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test

def scale_data(x_train, x_test):
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)

    return x_train, x_test