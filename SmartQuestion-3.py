# %%
# Smart Question :- 3

# Data visualization
sns.boxplot(x='fraud', y='distance_from_home', data=card_transdata)
plt.title('Comparison of Distance from Home for Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Fraudulent Transaction')
plt.ylabel('Distance from Home')
plt.show()

# Hypothesis test (Two-sample t-test)
t_stat, p_value = ttest_ind(card_transdata.loc[card_transdata['fraud'] == 0, 'distance_from_home'],
                            card_transdata.loc[card_transdata['fraud'] == 1, 'distance_from_home'],
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