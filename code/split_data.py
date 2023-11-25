import pandas as pd
from sklearn.model_selection import train_test_split

def get_split_normalized_data(dataset, ratio=0.8):
    features = dataset.drop(columns=["fraud"])
    labels = dataset["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ratio, random_state=0)

    mean_dist_from_home, sd_dist_from_home = X_train["distance_from_home"].mean(), X_train["distance_from_home"].sd()
    mean_dist_from_last, sd_dist_from_last = X_train["distance_from_last_transaction"].mean(), X_train["distance_from_last_transaction"].sd()
    mean_ratio_to_median, sd_ratio_to_median = X_train["ratio_to_median_purchase_price"].mean(), X_train["ratio_to_median_purchase_price"].sd() 

    X_train["distance_from_home"] = (X_train["distance_from_home"] - mean_dist_from_home) / sd_dist_from_home
    X_train["distance_from_last_transaction"] = ( X_train["distance_from_last_transaction"] - mean_dist_from_last) / sd_dist_from_last
    X_train["ratio_to_median_purchase_price"] = (X_train["ratio_to_median_purchase_price"] - mean_ratio_to_median) / sd_ratio_to_median

    X_test["distance_from_home"] = (X_test["distance_from_home"] - mean_dist_from_home) / sd_dist_from_home
    X_test["distance_from_last_transaction"] = ( X_test["distance_from_last_transaction"] - mean_dist_from_last) / sd_dist_from_last
    X_test["ratio_to_median_purchase_price"] = (X_test["ratio_to_median_purchase_price"] - mean_ratio_to_median) / sd_ratio_to_median

    return X_train, X_test, y_train, y_test