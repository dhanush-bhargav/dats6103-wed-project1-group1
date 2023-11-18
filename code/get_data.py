#%%
import pandas as pd

df = raw_data = pd.read_csv("/Users/dineshanandthulasiraman/Documents/GitHub/dats6103-wed-project1-group1/data/card_transdata.csv")

#%%
#Data Inspection
print(df.head())

#checking information of data
df.info()

#checking null values
df.isnull().sum()

#describing data
df.describe().transpose()

#checking 1st Row
df.iloc[0,:]
#%%
