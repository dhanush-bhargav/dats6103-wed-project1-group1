# %%
# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset (assuming 'card_transdata' is a DataFrame)
# pd.read_csv('your_dataset.csv')  # Uncomment and modify this line if reading from a CSV file
#%%
# Display summary statistics
card_transdata = pd.read_csv("your_dataset/card_transdata.csv")
print(card_transdata.describe())
print(card_transdata.info())
print(card_transdata.head())


# %%
# Smart Question :- 1

# Not Fraud in Online Orders (1)
count_not_fraud_online_order = card_transdata[(card_transdata['fraud'] == 0) & (card_transdata['online_order'] == 1)].shape[0]

# Not Fraud in Offline Orders (0)
count_not_fraud_offline_order = card_transdata[(card_transdata['fraud'] == 0) & (card_transdata['online_order'] == 0)].shape[0]

#Fraud in Online Orders (1)
count_fraud_online_order = card_transdata[(card_transdata['fraud'] == 1) & (card_transdata['online_order'] == 1)].shape[0]

#Fraud in Offline Orders (1)
count_fraud_offline_order = card_transdata[(card_transdata['fraud'] == 1) & (card_transdata['online_order'] == 0)].shape[0]

print("Not Fraud in Online Orders:", count_not_fraud_online_order)
print("Not Fraud in Offline Orders:", count_not_fraud_offline_order)
print("Fraud in Online Orders:", count_fraud_online_order)
print("Fraud in Offline Orders:", count_fraud_offline_order)

# Data visualization
colors = {'Offline Order': 'blue', 'Online Order': 'orange'}
sns.countplot(x='fraud', hue='online_order', data=card_transdata)
plt.title('Fraudulent Transactions by Type (Online/Offline)')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Count')
plt.legend(title='Order Type', labels=colors.keys())
plt.show()
#%%
# Hypothesis test (Chi-squared test for independence)
contingency_table = pd.crosstab(card_transdata['fraud'], card_transdata['online_order'])
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
