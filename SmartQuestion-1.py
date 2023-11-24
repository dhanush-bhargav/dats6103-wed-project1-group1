# %%
# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

# Load the dataset (assuming 'card_transdata' is a DataFrame)
# pd.read_csv('your_dataset.csv')  # Uncomment and modify this line if reading from a CSV file

# Display summary statistics
card_transdata = pd.read_csv(r"C:\Users\HP\Desktop\Class files\Data mining\Project\card_transdata.csv")
card_transdata.describe()

# %%
# Smart Question :- 1

# Data visualization
sns.countplot(x='fraud', hue='online_order', data=card_transdata)
plt.title('Fraudulent Transactions by Type (Online/Offline)')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Count')
plt.legend(title='Online Order')
plt.show()

# Hypothesis test (Chi-squared test for independence)
contingency_table = pd.crosstab(card_transdata['fraud'], card_transdata['online_order'])
chi2, p, _, _ = chi2_contingency(contingency_table)

print("H0: There is no association between the type of transaction (online or offline) and fraud.")
print("H1: There is an association between the type of transaction and fraud.")
print(f"Chi-squared test p-value: {p}")

if p < 0.05:
    print("As the p-value in the Chi-squared test is less than 0.05 (Standard Significance Level), we reject the Null Hypothesis.")
    print("Therefore, we accept the Alternate Hypothesis, which states that there is an association between the type of transaction and fraud.")
    print("From the visualization, it's clear that 'Online orders' are more commonly fraudulent.")
else:
    print("The p-value in the Chi-squared test is greater than or equal to 0.05, so we do not reject the Null Hypothesis.")
#%%