import split_data
import pandas as pd
import build_model

dataset = pd.read_csv("data/card_transdata.csv")
X_train, X_test, y_train, y_test = split_data.get_split_normalized_data(dataset)

lrcv_model = build_model.build_LR_model(X_train, y_train)
lrcv_preds = lrcv_model.predict(X_test)

knn_model1 = build_model.build_KNN_model(X_train, y_train, 10)
knn_preds = knn_model1.predict(X_test)

rf_model1 = build_model.build_forest_model(X_train, y_train, 100, 20)
rf_preds = rf_model1.predict(X_test)

svc_model = build_model.build_svc_model(X_train, y_train)
svc_preds = svc_model.predict(X_test)

#Calculate the model metrics (recall, confusion matrix, ROC-AUC, accuracy score).