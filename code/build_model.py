from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.formula.api import logit
import split_data
import pandas as pd


def build_LR_model(X_train, y_train):
    lr_model = LogisticRegressionCV(cv=10, scoring="recall", class_weight='balanced').fit(X_train, y_train) #class weights are balanced to compensate for imbalanced classes.
    return lr_model

def build_KNN_model(X_train, y_train, k):
    knn_model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    return knn_model

def build_forest_model(X_train, y_train, estimators, depth):
    RF_model = RandomForestClassifier(n_estimators=estimators, max_depth=depth).fit(X_train, y_train)
    return RF_model

def build_svc_model(X_train, y_train, kernel="linear"):
    svc_model = SVC(kernel=kernel, class_weight='balanced', probability=True).fit(X_train, y_train) #class weights are balanced to compensate for imbalanced classes.
    return svc_model
