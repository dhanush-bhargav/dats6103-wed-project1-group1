#%%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import split_data
import pandas as pd

import pandas as pd
from sklearn.model_selection import train_test_split

def get_split_normalized_data(dataset, ratio=0.8):
    features = dataset.drop(columns=["fraud"])
    labels = dataset["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ratio, random_state=0)

    mean_dist_from_home, sd_dist_from_home = X_train["distance_from_home"].mean(), X_train["distance_from_home"].std()
    mean_dist_from_last, sd_dist_from_last = X_train["distance_from_last_transaction"].mean(), X_train["distance_from_last_transaction"].std()
    mean_ratio_to_median, sd_ratio_to_median = X_train["ratio_to_median_purchase_price"].mean(), X_train["ratio_to_median_purchase_price"].std() 

    X_train["distance_from_home"] = (X_train["distance_from_home"] - mean_dist_from_home) / sd_dist_from_home
    X_train["distance_from_last_transaction"] = ( X_train["distance_from_last_transaction"] - mean_dist_from_last) / sd_dist_from_last
    X_train["ratio_to_median_purchase_price"] = (X_train["ratio_to_median_purchase_price"] - mean_ratio_to_median) / sd_ratio_to_median

    X_test["distance_from_home"] = (X_test["distance_from_home"] - mean_dist_from_home) / sd_dist_from_home
    X_test["distance_from_last_transaction"] = ( X_test["distance_from_last_transaction"] - mean_dist_from_last) / sd_dist_from_last
    X_test["ratio_to_median_purchase_price"] = (X_test["ratio_to_median_purchase_price"] - mean_ratio_to_median) / sd_ratio_to_median

    return X_train, X_test, y_train, y_test
#%%
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

#%%
if __name__=="__main__":
    dataset = pd.read_csv("/data/card_transdata.csv")
    X_train, X_test, y_train, y_test = split_data.get_split_normalized_data(dataset)
    lrcv_model = build_LR_model(X_train, y_train)
    
    knn_model1 = build_KNN_model(X_train, y_train, 10)
    knn_model2 = build_KNN_model(X_train, y_train, 100)
    knn_model3 = build_KNN_model(X_train, y_train, 250)

    rf_model1 = build_forest_model(X_train, y_train, 100, 20)
    rf_model2 = build_forest_model(X_train, y_train, 100, 10)

    rf_model3 = build_forest_model(X_train, y_train, 50, 20)
    rf_model4 = build_forest_model(X_train, y_train, 50, 10)

    svc_model = build_svc_model(X_train, y_train)

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

lrcv_accuracy, lrcv_precision, lrcv_recall, lrcv_f1 = evaluate_model(lrcv_model, X_test, y_test)


knn1_accuracy, knn1_precision, knn1_recall, knn1_f1 = evaluate_model(knn_model1, X_test, y_test)
knn2_accuracy, knn2_precision, knn2_recall, knn2_f1 = evaluate_model(knn_model2, X_test, y_test)
knn3_accuracy, knn3_precision, knn3_recall, knn3_f1 = evaluate_model(knn_model3, X_test, y_test)


rf1_accuracy, rf1_precision, rf1_recall, rf1_f1 = evaluate_model(rf_model1, X_test, y_test)
rf2_accuracy, rf2_precision, rf2_recall, rf2_f1 = evaluate_model(rf_model2, X_test, y_test)
rf3_accuracy, rf3_precision, rf3_recall, rf3_f1 = evaluate_model(rf_model3, X_test, y_test)
rf4_accuracy, rf4_precision, rf4_recall, rf4_f1 = evaluate_model(rf_model4, X_test, y_test)


svc_accuracy, svc_precision, svc_recall, svc_f1 = evaluate_model(svc_model, X_test, y_test)

# Print or use these metrics as needed
print("Logistic Regression - Accuracy:", lrcv_accuracy, "Precision:", lrcv_precision, "Recall:", lrcv_recall, "F1-score:", lrcv_f1)
# Similarly, print or use other models' metrics

# %%

print("Logistic Regression - Accuracy:", lrcv_accuracy, "Precision:", lrcv_precision, "Recall:", lrcv_recall, "F1-score:", lrcv_f1)


print("KNN Model 1 - Accuracy:", knn1_accuracy, "Precision:", knn1_precision, "Recall:", knn1_recall, "F1-score:", knn1_f1)
print("KNN Model 2 - Accuracy:", knn2_accuracy, "Precision:", knn2_precision, "Recall:", knn2_recall, "F1-score:", knn2_f1)
print("KNN Model 3 - Accuracy:", knn3_accuracy, "Precision:", knn3_precision, "Recall:", knn3_recall, "F1-score:", knn3_f1)


print("Random Forest Model 1 - Accuracy:", rf1_accuracy, "Precision:", rf1_precision, "Recall:", rf1_recall, "F1-score:", rf1_f1)
print("Random Forest Model 2 - Accuracy:", rf2_accuracy, "Precision:", rf2_precision, "Recall:", rf2_recall, "F1-score:", rf2_f1)
print("Random Forest Model 3 - Accuracy:", rf3_accuracy, "Precision:", rf3_precision, "Recall:", rf3_recall, "F1-score:", rf3_f1)
print("Random Forest Model 4 - Accuracy:", rf4_accuracy, "Precision:", rf4_precision, "Recall:", rf4_recall, "F1-score:", rf4_f1)

print("SVC - Accuracy:", svc_accuracy, "Precision:", svc_precision, "Recall:", svc_recall, "F1-score:", svc_f1)

# %%

# time pass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
lrcv_model_predictions = lrcv_model.predict(X_test)
r_squared = r2_score(y_test, lrcv_model_predictions)
rmse = mean_squared_error(y_test, lrcv_model_predictions, squared=False)

print("Regression - R-squared:", r_squared)
print("Regression - RMSE:", rmse)

# %%

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to plot confusion matrices
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Get confusion matrices for each model
conf_matrices = {}

# Logistic Regression
lrcv_cm = confusion_matrix(y_test, lrcv_model.predict(X_test))
conf_matrices['Logistic Regression'] = lrcv_cm
plot_confusion_matrix(lrcv_cm, ['Non-Fraud', 'Fraud'])

# KNN Models
knn_cm1 = confusion_matrix(y_test, knn_model1.predict(X_test))
conf_matrices['KNN (k=10)'] = knn_cm1
plot_confusion_matrix(knn_cm1, ['Non-Fraud', 'Fraud'])

knn_cm2 = confusion_matrix(y_test, knn_model2.predict(X_test))
conf_matrices['KNN (k=100)'] = knn_cm2
plot_confusion_matrix(knn_cm2, ['Non-Fraud', 'Fraud'])

# Repeat for other models...

# Random Forest Models
rf_cm1 = confusion_matrix(y_test, rf_model1.predict(X_test))
conf_matrices['Random Forest (100, 20)'] = rf_cm1
plot_confusion_matrix(rf_cm1, ['Non-Fraud', 'Fraud'])

rf_cm2 = confusion_matrix(y_test, rf_model2.predict(X_test))
conf_matrices['Random Forest (100, 10)'] = rf_cm2
plot_confusion_matrix(rf_cm2, ['Non-Fraud', 'Fraud'])

# Repeat for other models...

# Support Vector Machine
svc_cm = confusion_matrix(y_test, svc_model.predict(X_test))
conf_matrices['Support Vector Machine'] = svc_cm
plot_confusion_matrix(svc_cm, ['Non-Fraud', 'Fraud'])

# Access confusion matrices using conf_matrices dictionary (e.g., conf_matrices['Logistic Regression'])


# %%

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probabilities for each model
lrcv_probs = lrcv_model.predict_proba(X_test)[:, 1]
fpr_lrcv, tpr_lrcv, _ = roc_curve(y_test, lrcv_probs)
auc_lrcv = roc_auc_score(y_test, lrcv_probs)

# Plot ROC curve for Logistic Regression
plt.figure(figsize=(8, 6))
plt.plot(fpr_lrcv, tpr_lrcv, label='Logistic Regression (AUC = %0.2f)' % auc_lrcv)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()

# Repeat for other models
# For instance, for KNN with k=10
knn_probs = knn_model1.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
auc_knn = roc_auc_score(y_test, knn_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label='KNN (k=10) (AUC = %0.2f)' % auc_knn)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN (k=10)')
plt.legend()
plt.show()

# Repeat for other models...
#%%
# KNN Model 2
knn_probs2 = knn_model2.predict_proba(X_test)[:, 1]
fpr_knn2, tpr_knn2, _ = roc_curve(y_test, knn_probs2)
auc_knn2 = roc_auc_score(y_test, knn_probs2)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn2, tpr_knn2, label='KNN (k=100) (AUC = %0.2f)' % auc_knn2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN (k=100)')
plt.legend()
plt.show()

# KNN Model 3
knn_probs3 = knn_model3.predict_proba(X_test)[:, 1]
fpr_knn3, tpr_knn3, _ = roc_curve(y_test, knn_probs3)
auc_knn3 = roc_auc_score(y_test, knn_probs3)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn3, tpr_knn3, label='KNN (k=250) (AUC = %0.2f)' % auc_knn3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN (k=250)')
plt.legend()
plt.show()


# %%
# Random Forest Model 1
rf_probs1 = rf_model1.predict_proba(X_test)[:, 1]
fpr_rf1, tpr_rf1, _ = roc_curve(y_test, rf_probs1)
auc_rf1 = roc_auc_score(y_test, rf_probs1)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf1, tpr_rf1, label='Random Forest (100, 20) (AUC = %0.2f)' % auc_rf1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (100, 20)')
plt.legend()
plt.show()

# Random Forest Model 2
rf_probs2 = rf_model2.predict_proba(X_test)[:, 1]
fpr_rf2, tpr_rf2, _ = roc_curve(y_test, rf_probs2)
auc_rf2 = roc_auc_score(y_test, rf_probs2)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf2, tpr_rf2, label='Random Forest (100, 10) (AUC = %0.2f)' % auc_rf2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (100, 10)')
plt.legend()
plt.show()

# %%
