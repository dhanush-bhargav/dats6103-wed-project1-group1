# %%
# Smart Question :- 3
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
#card_transdata = pd.read_csv(r"C:\Users\HP\Desktop\Class files\Data mining\Project\card_transdata.csv")
card_transdata = pd.read_csv("your_dataset/card_transdata.csv")

#
# Data visualization
sns.boxplot(x='fraud', y='distance_from_home', data=card_transdata)
plt.title('Comparison of Distance from Home for Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Distance from Home')
plt.show()
data=card_transdata

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
t_stat, p_value = ttest_ind(card_transdata.loc[card_transdata['fraud'] == 0, 'distance_from_home'],
                            card_transdata.loc[card_transdata['fraud'] == 1, 'distance_from_home'],
                            equal_var=False)

print("H0: There is no difference in the mean distance from home between fraudulent and non-fraudulent transactions.")
print("H1: There is a significant difference in the mean distance from home between fraudulent and non-fraudulent transactions.")
print(f"Two-sample t-test p-value: {p_value}")

if p_value < 0.05:
    print("As the p-value in the two-sample t-test is less than 0.05 (Standard Significance Level), we reject the Null Hypothesis.")
    print("Therefore, we accept the Alternate Hypothesis, which states that there is a significant difference in the\nmean distance from home between fraudulent and non-fraudulent transactions.")
    print("This means that a correlation exists between the distance from home where a transaction occurred and whether the transaction was fraudulent.")
else:
    print("The p-value in the two-sample t-test is greater than or equal to 0.05, so we do not reject the Null Hypothesis.")

# %%
