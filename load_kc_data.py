# Load and clean the King County housing prices data set

import pandas as pd
import numpy as np

def load_kc_housing():
    king_county_df = pd.read_csv("kc_house_data.csv")
    king_county_df = king_county_df[~king_county_df["id"].duplicated()] # Get rid of some duplicate rows
    king_county_df["bedrooms"][15870] = 3 # This is the most glaring outlier value. Making my best guess as to what it should be
    # Split into training and test
    np.random.seed(12345)
    rand_nums = np.random.rand(len(king_county_df))
    msk_train = rand_nums < 0.6
    msk_dev = (rand_nums >= 0.6) & (rand_nums < 0.8)
    msk_test = rand_nums >= 0.8
    train_df = king_county_df[msk_train]
    dev_df = king_county_df[msk_dev]
    test_df = king_county_df[msk_test]
    
    # Restrict the features
    # The following are excluded: lat, long, id, date.
    # Also excluding sqft_living, as this is the sum of sqft_above and sqft_basement
    # For simplicity, dropping zipcode too.
    features = [u'bedrooms', u'bathrooms', u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade', 
        u'sqft_above', u'sqft_basement', u'yr_built', u'yr_renovated', u'sqft_living15', u'sqft_lot15']
    train_X = train_df[features]
    dev_X = dev_df[features]
    test_X = test_df[features]
    train_y = train_df["price"]
    dev_y = dev_df["price"]
    test_y = test_df["price"]
    
    return train_X, train_y, dev_X, dev_y, test_X, test_y