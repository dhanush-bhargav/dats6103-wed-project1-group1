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

def build_svc_model(X_train, y_train):
    svc_model = SVC(kernel='linear', class_weight='balanced').fit(X_train, y_train) #class weights are balanced to compensate for imbalanced classes.
    return svc_model


if __name__=="__main__":
    dataset = pd.read_csv("data/card_transdata.csv")
    X_train, X_test, y_train, y_test = split_data.get_split_normalized_data(dataset)
    # lrcv_model = build_LR_model(X_train, y_train)
    
    # knn_model1 = build_KNN_model(X_train, y_train, 10)
    # knn_model2 = build_KNN_model(X_train, y_train, 100)
    # knn_model3 = build_KNN_model(X_train, y_train, 250)

    # rf_model1 = build_forest_model(X_train, y_train, 100, 20)
    # rf_model2 = build_forest_model(X_train, y_train, 100, 10)

    # rf_model3 = build_forest_model(X_train, y_train, 50, 20)
    # rf_model4 = build_forest_model(X_train, y_train, 50, 10)

    # svc_model = build_svc_model(X_train, y_train)

    logit_model = logit("fraud ~ distance_from_home + distance_from_last_transaction + ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order", dataset).fit()
    print(logit_model.summary())