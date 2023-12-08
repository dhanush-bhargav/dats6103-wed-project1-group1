from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_LR_model(X_train, y_train):
    lr_model = LogisticRegressionCV(cv=10, scoring="recall").fit(X_train, y_train)
    return lr_model

def build_KNN_model(X_train, y_train, k):
    knn_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return knn_model

def build_forest_model(X_train, y_train, estimators, depth):
    RF_model = RandomForestClassifier(n_estimators=estimators, max_depth=depth).fit(X_train, y_train)
    return RF_model

def build_svc_model(X_train, y_train):
    svc_model = SVC(kernel='linear').fit(X_train, y_train)
    return svc_model
