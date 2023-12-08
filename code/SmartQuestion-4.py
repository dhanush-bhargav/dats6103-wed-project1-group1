# %%
# Smart Question :- 4
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
#card_transdata = pd.read_csv("/Users/manojpadala/Desktop/git/dats6103-wed-project1-group1/data/card_transdata.csv")
card_transdata = pd.read_csv('your_dataset.csv')
# Data visualization
sns.boxplot(x='fraud', y='distance_from_last_transaction', data=card_transdata)
plt.title('Distance from last transaction and the transaction being reported as fraud')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Distance from Last Transaction')
plt.show()

# Hypothesis test (Two-sample t-test)
t_stat, p_value = ttest_ind(card_transdata.loc[card_transdata['fraud'] == 0, 'distance_from_last_transaction'],
                            card_transdata.loc[card_transdata['fraud'] == 1, 'distance_from_last_transaction'],
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