#%%
import pandas as pd

df = raw_data = pd.read_csv("/Users/dineshanandthulasiraman/Documents/GitHub/dats6103-wed-project1-group1/data/card_transdata.csv")

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



