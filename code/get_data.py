import pandas as pd

raw_data = pd.read_csv("data/card_transdata.csv")

if raw_data.isna().sum().sum() == 0:
    print("No missing data in the dataset")

print(raw_data.median())