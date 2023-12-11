#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

#%%
from build_model import *
from split_data import get_split_normalized_data
from model_metrics import *

#%%
df = pd.read_csv("../data/card_transdata.csv")

#%%
#Data Inspection
print(df.head())

#%%
#checking information of data
df.info()

#%%
#checking null values
df.isnull().sum()

#%%
#describing data
df.describe().transpose()

# %%

# Visualize correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# %%
# Visualize the distribution of the target variable
sns.countplot(x='fraud', data=df)
plt.show()

#%%
# Features to visualize
selected_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

for feature in selected_features:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[feature], ax=ax, kde=True, bins=30)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    plt.show()
    descriptive_stats = df[feature].describe()
    print(f'Descriptive Statistics for {feature}:\n{descriptive_stats}\n{"="*40}\n')

# %%
# Features to visualize
selected_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']

for feature in selected_features:
    # Rescale the feature to a range not including zero
    rescaled_feature = (df[feature] - df[feature].min() + 1)  # Adding 1 to avoid log(0)
    
    # Apply log transform to the rescaled feature
    log_transformed_feature = np.log(rescaled_feature)
    
    # Create violin plot with log-transformed feature
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='fraud', y=log_transformed_feature, data=df, aspect=2)
    plt.title(f'Violin Plot for {feature}')
    plt.xlabel('Fraud')
    plt.ylabel(feature)
    plt.show()

# %%
# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

#%%

# Relationship between 'distance_from_home' and 'distance_from_last_transaction'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='distance_from_home', y='distance_from_last_transaction', hue='fraud', data=df)

plt.xlim(0, 1000)  
plt.ylim(0, 500)  

plt.title('Relationship between Distance from Home and Last Transaction')
plt.xlabel('Distance from Home')
plt.ylabel('Distance from Last Transaction')
plt.show()

#%%
# Analysis of 'ratio_to_median_purchase_price'
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='fraud', y='ratio_to_median_purchase_price', data=df)
plt.title('Boxplot of Ratio to Median Purchase Price')

plt.subplot(1, 2, 2)
sns.histplot(df['ratio_to_median_purchase_price'], bins=30, kde=True)
plt.title('Distribution of Ratio to Median Purchase Price')
plt.xlabel('Ratio to Median Purchase Price')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
# Analysis of 'repeat_retailer' and 'fraud'
sns.countplot(x='repeat_retailer', hue='fraud', data=df)
plt.title('Count of Fraudulent Transactions based on Repeated Retailer')
plt.xlabel('Repeat Retailer')
plt.show()

#%%
# Pairplot with a focus on 'fraud'
sns.pairplot(df, hue='fraud', vars=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price'])
plt.suptitle('Pairplot')
plt.show()


# %%
# Smart Question :- 1

# Not Fraud in Online Orders (1)
count_not_fraud_online_order = df[(df['fraud'] == 0) & (df['online_order'] == 1)].shape[0]

# Not Fraud in Offline Orders (0)
count_not_fraud_offline_order = df[(df['fraud'] == 0) & (df['online_order'] == 0)].shape[0]

#Fraud in Online Orders (1)
count_fraud_online_order = df[(df['fraud'] == 1) & (df['online_order'] == 1)].shape[0]

#Fraud in Offline Orders (1)
count_fraud_offline_order = df[(df['fraud'] == 1) & (df['online_order'] == 0)].shape[0]

print("Not Fraud in Online Orders:", count_not_fraud_online_order)
print("Not Fraud in Offline Orders:", count_not_fraud_offline_order)
print("Fraud in Online Orders:", count_fraud_online_order)
print("Fraud in Offline Orders:", count_fraud_offline_order)

# Data visualization
colors = {'Offline Order': 'blue', 'Online Order': 'orange'}
sns.countplot(x='fraud', hue='online_order', data=df)
plt.title('Fraudulent Transactions by Type (Online/Offline)')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Count')
plt.legend(title='Order Type', labels=colors.keys())
plt.show()

#%%
# Hypothesis test (Chi-squared test for independence)
contingency_table = pd.crosstab(df['fraud'], df['online_order'])
chi2, p, _, _ = chi2_contingency(contingency_table)

print("H0: There is no associaticon between the type of transaction (online or offline) and fraud.")
print("H1: There is an association between the type of transaction and fraud.")
print(f"Chi-squared test p-value: {p}")

if p < 0.05:
    print("As the p-value in the Chi-squared test is less than 0.05 (Standard Significance Level), we reject the Null Hypothesis.")
    print("Therefore, we accept the Alternate Hypothesis, which states that there is an association between the type of transaction and fraud.")
    print("From the visualization, it's clear that 'Online orders' are more commonly fraudulent.")
else:
    print("The p-value in the Chi-squared test is greater than or equal to 0.05, so we do not reject the Null Hypothesis.")

# %%
#
# Data visualization
sns.boxplot(x='fraud', y='distance_from_home', data=df)
plt.title('Comparison of Distance from Home for Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Distance from Home')
plt.show()
data=df

bins_home = [0, 10, 100, 500, 1000, 4000, 8000, 15000]

bin_labels_home = ['0-10', '10-100', '100-500', '500-1000', '1000-4000', '4000-8000', '8000-15000']

data['range_home'] = pd.cut(data['distance_from_home'], bins=bins_home, labels=bin_labels_home)

grouped_home = data.groupby(['fraud', 'range_home']).size().unstack()


grouped_home.T.plot(kind='line', marker='o', figsize=(10, 6))
plt.xlabel('Range of Distance from Home')
plt.ylabel('Count of Distance')
plt.title('Count of Distance from Home by Range')
plt.xticks(rotation=45)
plt.legend(title='Fraud', loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
import pandas as pd

bins = [0, 10, 100, 500, 1000, 4000, 8000, 15000]
bin_labels = ['0-10', '10-100', '100-500', '500-1000', '1000-4000', '4000-8000', '8000-15000']

data['range'] = pd.cut(data['distance_from_home'], bins=bins, labels=bin_labels)
grouped = data.groupby(['fraud', 'range']).size().unstack().fillna(0).astype(int)

print("Table of Count of Distance from Home by Range and Fraud")
print(grouped)

#%%
# Hypothesis test (Two-sample t-test)
t_stat, p_value = ttest_ind(df.loc[df['fraud'] == 0, 'distance_from_home'],
                            df.loc[df['fraud'] == 1, 'distance_from_home'],
                            equal_var=False)

print("H0: There is no difference in the mean distance from home between fraudulent and non-fraudulent transactions.")
print("H1: There is a significant difference in the mean distance from home between fraudulent and non-fraudulent transactions.")
print(f"Two-sample t-test p-value: {p_value}")

if p_value < 0.05:
    print("As the p-value in the two-sample t-test is less than 0.05 (Standard Significance Level), we reject the Null Hypothesis.")
    print("Therefore, we accept the Alternate Hypothesis, which states that there is a significant difference in the mean distance from home between fraudulent and non-fraudulent transactions.")
    print("This means that a correlation exists between the distance from home where a transaction occurred and whether the transaction was fraudulent.")
else:
    print("The p-value in the two-sample t-test is greater than or equal to 0.05, so we do not reject the Null Hypothesis.")


# %%
data=df
bins = [0, 10, 100, 500, 1000, 4000, 8000, 15000]

bin_labels = ['0-10', '10-100', '100-500', '500-1000', '1000-4000', '4000-8000', '8000-15000']

data['range'] = pd.cut(data['distance_from_last_transaction'], bins=bins, labels=bin_labels)

grouped = data.groupby(['fraud', 'range']).size().unstack()

grouped.T.plot(kind='line', marker='o', figsize=(10, 6))
plt.xlabel('Range of Distance')
plt.ylabel('Count of Distance')
plt.title('Count of Distance from Last Transaction by Range')
plt.xticks(rotation=45)
plt.legend(title='Fraud', loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
bins = [0, 10, 100, 500, 1000, 4000, 8000, 15000]
bin_labels = ['0-10', '10-100', '100-500', '500-1000', '1000-4000', '4000-8000', '8000-15000']
data['range'] = pd.cut(data['distance_from_last_transaction'], bins=bins, labels=bin_labels)
grouped = data.groupby(['fraud', 'range']).size().unstack().fillna(0).astype(int)
print("Table of Count of Distance from Last Transaction by Range and Fraud")
print(grouped)

# %%
# Data visualization
sns.boxplot(x='fraud', y='distance_from_home', data=df)
plt.title('Comparison of Distance from Home for Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Distance from Home')
plt.show()

# Hypothesis test (Two-sample t-test)
t_stat, p_value = ttest_ind(df.loc[df['fraud'] == 0, 'distance_from_last_transaction'],
                            df.loc[df['fraud'] == 1, 'distance_from_last_transaction'],
                            equal_var=False)

print("H0: There is no difference in the mean distance from the last transaction between fraudulent and non-fraudulent transactions.")
print("H1: There is a significant difference in the mean distance from the last transaction between fraudulent and non-fraudulent transactions.")
print(f"Two-sample t-test p-value: {p_value}")

if p_value < 0.05:
    print("As the p-value in the two-sample t-test is less than 0.05 (Standard Significance Level), we reject the Null Hypothesis.")
    print("Therefore, we accept the Alternate Hypothesis, which states that there is a significant difference in the mean distance from the last transaction between fraudulent and non-fraudulent transactions.")
    print("This means that a correlation exists between the distance from the last transaction and the transaction being reported as fraud.")
else:
    print("The p-value in the two-sample t-test is greater than or equal to 0.05, so we do not reject the Null Hypothesis.")


#%%
X_train, X_test, y_train, y_test = get_split_normalized_data(df)

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