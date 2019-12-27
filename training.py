import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2000


def load_clean_data():
    return pd.read_csv("clean_data.csv", index_col=0, encoding="latin-1")


def split_clean_data():
    """
    Split clean data into training and validation set.

    Returns:
    x_train, x_validation, y_train, y_validation
    """
    data = load_clean_data()
    return train_test_split(
        data.SentimentText, data.Sentiment, test_size=0.2, random_state=SEED
    )
