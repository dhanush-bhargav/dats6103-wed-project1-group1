#%%
import pandas as pd
from build_model import *
from split_data import get_split_normalized_data
from model_metrics import *

#%%
dataset = pd.read_csv("../data/card_transdata.csv")

#%%

X_train, X_test, y_train, y_test = get_split_normalized_data(dataset)

#%%

lrcv_model = build_LR_model(X_train, y_train)

y_pred_lrcv = lrcv_model.predict_proba(X_train).T[1]
lrcv_accuracy, lrcv_precision, lrcv_recall, lrcv_f1 = evaluate_model(y_train, y_pred_lrcv, cutoff=0.5)
print("Logistic Regression Training metrics - Accuracy:", lrcv_accuracy, "Precision:", lrcv_precision, "Recall:", lrcv_recall, "F1-score:", lrcv_f1)

#%%
knn_model1 = build_KNN_model(X_train, y_train, 10)

y_pred_knn1 = knn_model1.predict_proba(X_train).T[1]
knn1_accuracy, knn1_precision, knn1_recall, knn1_f1 = evaluate_model(y_train, y_pred_knn1, cutoff=0.5)
print("KNN Model 1 Training metrics - Accuracy:", knn1_accuracy, "Precision:", knn1_precision, "Recall:", knn1_recall, "F1-score:", knn1_f1)

knn_model2 = build_KNN_model(X_train, y_train, 100)

y_pred_knn2 = knn_model2.predict_proba(X_train).T[1]
knn2_accuracy, knn2_precision, knn2_recall, knn2_f1 = evaluate_model(y_train, y_pred_knn2, cutoff=0.5)
print("KNN Model 2 Training metrics - Accuracy:", knn2_accuracy, "Precision:", knn2_precision, "Recall:", knn2_recall, "F1-score:", knn2_f1)

knn_model3 = build_KNN_model(X_train, y_train, 250)

y_pred_knn3 = knn_model3.predict_proba(X_train).T[1]
knn3_accuracy, knn3_precision, knn3_recall, knn3_f1 = evaluate_model(y_train, y_pred_knn3, cutoff=0.5)
print("KNN Model 3 training metrics - Accuracy:", knn3_accuracy, "Precision:", knn3_precision, "Recall:", knn3_recall, "F1-score:", knn3_f1)

#%%
rf_model1 = build_forest_model(X_train, y_train, 100, 20)

y_pred_rf1 = rf_model1.predict_proba(X_train).T[1]
rf1_accuracy, rf1_precision, rf1_recall, rf1_f1 = evaluate_model(y_train, y_pred_rf1, cutoff=0.5)
print("Random Forest Model 1 Training metrics - Accuracy:", rf1_accuracy, "Precision:", rf1_precision, "Recall:", rf1_recall, "F1-score:", rf1_f1)

rf_model2 = build_forest_model(X_train, y_train, 100, 10)

y_pred_rf2 = rf_model2.predict_proba(X_train).T[1]
rf2_accuracy, rf2_precision, rf2_recall, rf2_f1 = evaluate_model(y_train, y_pred_rf2, cutoff=0.5)
print("Random Forest Model 2 Training metrics - Accuracy:", rf2_accuracy, "Precision:", rf2_precision, "Recall:", rf2_recall, "F1-score:", rf2_f1)

rf_model3 = build_forest_model(X_train, y_train, 50, 20)

y_pred_rf3 = rf_model3.predict_proba(X_train).T[1]
rf3_accuracy, rf3_precision, rf3_recall, rf3_f1 = evaluate_model(y_train, y_pred_rf3, cutoff=0.5)
print("Random Forest Model 3 Training metrics - Accuracy:", rf3_accuracy, "Precision:", rf3_precision, "Recall:", rf3_recall, "F1-score:", rf3_f1)

rf_model4 = build_forest_model(X_train, y_train, 50, 10)

y_pred_rf4 = rf_model4.predict_proba(X_train).T[1]
rf4_accuracy, rf4_precision, rf4_recall, rf4_f1 = evaluate_model(y_train, y_pred_rf4, cutoff=0.5)
print("Random Forest Model 4 Training metrics - Accuracy:", rf4_accuracy, "Precision:", rf4_precision, "Recall:", rf4_recall, "F1-score:", rf4_f1)

#%%
svc_model_linear = build_svc_model(X_train, y_train, kernel="linear")

y_pred_svc_linear = svc_model_linear.predict_proba(X_train).T[1]
svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(y_train, y_pred_svc_linear, cutoff=0.5)
print("SVC Linear kernel Training metrics - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

svc_model_rbf = build_svc_model(X_train, y_train, kernel="rbf")

y_pred_svc_rbf = svc_model_rbf.predict_proba(X_train).T[1]
svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(y_train, y_pred_svc_rbf, cutoff=0.5)
print("SVC RBF kernel Training metrics - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

#%%
y_pred_lrcv = lrcv_model.predict_proba(X_test).T[1]
lrcv_accuracy, lrcv_precision, lrcv_recall, lrcv_f1 = evaluate_model(y_test, y_pred_lrcv, cutoff=0.5)
print("Logistic Regression Test metrics - Accuracy:", lrcv_accuracy, "Precision:", lrcv_precision, "Recall:", lrcv_recall, "F1-score:", lrcv_f1)

plot_confusion_matrix(y_test, y_pred_lrcv, labels=['No fraud', 'Fraud'], model_name="Logistic Regression", cutoff=0.5)

#%%
y_pred_knn1 = knn_model1.predict_proba(X_test).T[1]
knn1_accuracy, knn1_precision, knn1_recall, knn1_f1 = evaluate_model(y_test, y_pred_knn1, cutoff=0.5)
print("KNN Model 1 Training metrics - Accuracy:", knn1_accuracy, "Precision:", knn1_precision, "Recall:", knn1_recall, "F1-score:", knn1_f1)

plot_confusion_matrix(y_test, y_pred_knn1, labels=['No fraud', 'Fraud'], model_name="KNN Model 1", cutoff=0.5)

#%%
y_pred_knn2 = knn_model2.predict_proba(X_test).T[1]
knn2_accuracy, knn2_precision, knn2_recall, knn2_f1 = evaluate_model(y_test, y_pred_knn2, cutoff=0.5)
print("KNN Model 2 Training metrics - Accuracy:", knn2_accuracy, "Precision:", knn2_precision, "Recall:", knn2_recall, "F1-score:", knn2_f1)

plot_confusion_matrix(y_test, y_pred_knn2, labels=['No fraud', 'Fraud'], model_name="KNN Model 2", cutoff=0.5)

#%%
y_pred_knn3 = knn_model3.predict_proba(X_test).T[1]
knn3_accuracy, knn3_precision, knn3_recall, knn3_f1 = evaluate_model(y_test, y_pred_knn3, cutoff=0.5)
print("KNN Model 3 training metrics - Accuracy:", knn3_accuracy, "Precision:", knn3_precision, "Recall:", knn3_recall, "F1-score:", knn3_f1)

plot_confusion_matrix(y_test, y_pred_knn3, labels=['No fraud', 'Fraud'], model_name="KNN Model 2", cutoff=0.5)

#%%
y_pred_rf1 = rf_model1.predict_proba(X_test).T[1]
rf1_accuracy, rf1_precision, rf1_recall, rf1_f1 = evaluate_model(y_test, y_pred_rf1, cutoff=0.5)
print("Random Forest Model 1 Training metrics - Accuracy:", rf1_accuracy, "Precision:", rf1_precision, "Recall:", rf1_recall, "F1-score:", rf1_f1)

plot_confusion_matrix(y_test, y_pred_rf1, labels=['No fraud', 'Fraud'], model_name="Random Forest 1", cutoff=0.5)

#%%
y_pred_rf2 = rf_model2.predict_proba(X_test).T[1]
rf2_accuracy, rf2_precision, rf2_recall, rf2_f1 = evaluate_model(y_test, y_pred_rf2, cutoff=0.5)
print("Random Forest Model 2 Training metrics - Accuracy:", rf2_accuracy, "Precision:", rf2_precision, "Recall:", rf2_recall, "F1-score:", rf2_f1)

plot_confusion_matrix(y_test, y_pred_rf2, labels=['No fraud', 'Fraud'], model_name="Random Forest 2", cutoff=0.5)

#%%
y_pred_rf3 = rf_model3.predict_proba(X_test).T[1]
rf3_accuracy, rf3_precision, rf3_recall, rf3_f1 = evaluate_model(y_test, y_pred_rf3, cutoff=0.5)
print("Random Forest Model 3 Training metrics - Accuracy:", rf3_accuracy, "Precision:", rf3_precision, "Recall:", rf3_recall, "F1-score:", rf3_f1)

plot_confusion_matrix(y_test, y_pred_rf3, labels=['No fraud', 'Fraud'], model_name="Random Forest 3", cutoff=0.5)

#%%
y_pred_rf4 = rf_model4.predict_proba(X_test).T[1]
rf4_accuracy, rf4_precision, rf4_recall, rf4_f1 = evaluate_model(y_test, y_pred_rf4, cutoff=0.5)
print("Random Forest Model 4 Training metrics - Accuracy:", rf4_accuracy, "Precision:", rf4_precision, "Recall:", rf4_recall, "F1-score:", rf4_f1)

plot_confusion_matrix(y_test, y_pred_rf4, labels=['No fraud', 'Fraud'], model_name="Random Forest 4", cutoff=0.5)

#%%
y_pred_svc_linear = svc_model_linear.predict_proba(X_test).T[1]
svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(y_test, y_pred_svc_linear, cutoff=0.5)
print("SVC Linear kernel Training metrics - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

plot_confusion_matrix(y_test, y_pred_svc_linear, labels=['No fraud', 'Fraud'], model_name="SVC Linear", cutoff=0.5)

#%%
y_pred_svc_rbf = svc_model_rbf.predict_proba(X_test).T[1]
svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(y_test, y_pred_svc_rbf, cutoff=0.5)
print("SVC RBF kernel Training metrics - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

plot_confusion_matrix(y_test, y_pred_svc_rbf, labels=['No fraud', 'Fraud'], model_name="SVC RBF", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_lrcv, model_name="Logistic Regression", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_knn1, model_name="KNN Model 1", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_knn2, model_name="KNN Model 2", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_knn3, model_name="KNN Model 3", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_rf1, model_name="Random Forest 1", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_rf2, model_name="Random Forest 2", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_rf3, model_name="Random Forest 3", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_rf4, model_name="Random Forest 4", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_svc_linear, model_name="SVC Linear", cutoff=0.5)

#%%
plot_roc_auc(y_test, y_pred_svc_rbf, model_name="SVC RBF", cutoff=0.5)