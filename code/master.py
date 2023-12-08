#%%
import split_data
import pandas as pd
import build_model
from model_metrics import evaluate_model, plot_confusion_matrix, plot_roc_auc
from statsmodels.formula.api import logit

#%%
dataset = pd.read_csv("../data/card_transdata.csv")
X_train, X_test, y_train, y_test = split_data.get_split_normalized_data(dataset)

#%%
logit_model = logit("fraud ~ distance_from_home + distance_from_last_transaction + ratio_to_median_purchase_price + repeat_retailer + used_chip + used_pin_number + online_order", dataset).fit()
print(logit_model.summary())

#%%
lrcv_model = build_model.build_LR_model(X_train, y_train)
knn_model1 = build_model.build_KNN_model(X_train, y_train, 10)
rf_model1 = build_model.build_forest_model(X_train, y_train, 100, 20)
svc_model = build_model.build_svc_model(X_train, y_train)

#%%
y_pred_lrcv = lrcv_model.predict_proba(X_test).T[1]
lrcv_accuracy, lrcv_precision, lrcv_recall, lrcv_f1 = evaluate_model(y_test, y_pred_lrcv, cutoff=0.5)
print("Logistic Regression - Accuracy:", lrcv_accuracy, "Precision:", lrcv_precision, "Recall:", lrcv_recall, "F1-score:", lrcv_f1)

#%%
y_pred_knn = knn_model1.predict_proba(X_test).T[1]
knn1_accuracy, knn1_precision, knn1_recall, knn1_f1 = evaluate_model(y_test, y_pred_knn, cutoff=0.5)
print("KNN Model 1 - Accuracy:", knn1_accuracy, "Precision:", knn1_precision, "Recall:", knn1_recall, "F1-score:", knn1_f1)

#%%
y_pred_rf = rf_model1.predict_proba(X_test).T[1]
rf1_accuracy, rf1_precision, rf1_recall, rf1_f1 = evaluate_model(y_test, y_pred_rf, cutoff=0.5)
print("Random Forest Model 1 - Accuracy:", rf1_accuracy, "Precision:", rf1_precision, "Recall:", rf1_recall, "F1-score:", rf1_f1)

#%%
y_pred_svc = svc_model.predict_proba(X_test).T[1]
svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(y_test, y_pred_svc, cutoff=0.5)
print("SVC - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

#%%
plot_confusion_matrix(y_test, y_pred_lrcv, labels=['No fraud', 'Fraud'], cutoff=0.5)

#%%
plot_confusion_matrix(y_test, y_pred_knn, labels=['No fraud', 'Fraud'], cutoff=0.5)

#%%
plot_confusion_matrix(y_test, y_pred_rf, labels=['No fraud', 'Fraud'], cutoff=0.5)

#%%
plot_confusion_matrix(y_test, y_pred_svc, labels=['No fraud', 'Fraud'], cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_lrcv, model_name="Logistic Regression", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_knn, model_name="KNN with K=10", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_rf, model_name="Random Forest (100, 20)", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_svc, model_name="Support Vector Classifier", cutoff=0.5)