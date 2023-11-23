#%%
import pandas as pd

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

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# Visualize relationships between numerical variables
sns.pairplot(df, hue='fraud', diag_kind='kde')
plt.show()
# %%

# Visualize correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# %%
# Visualize the distribution of the target variable
sns.countplot(x='fraud', data=df)
plt.show()
# %%
# Visualize the distribution of numerical features
num_features = df.select_dtypes(include=[np.number]).columns
for feature in num_features:
    sns.histplot(df[feature])
    plt.title(f'Distribution of {feature}')
    plt.show()

# %%
# Visualize boxplots for numerical features to identify outliers
for feature in num_features:
    sns.boxplot(x='fraud', y=feature, data=df)
    plt.title(f'Boxplot for {feature}')
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
plt.suptitle('Pairplot with a Focus on Fraud')
plt.show()

# %%
