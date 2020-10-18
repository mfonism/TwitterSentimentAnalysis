import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLEAN_DATA_FILE,
    RANDOM_STATE,
    TRAIN_X_FILE,
    TEST_X_FILE,
    TRAIN_Y_FILE,
    TEST_Y_FILE,
)

df = pd.read_csv(CLEAN_DATA_FILE)
df_nonhate = df[df["Polarity"] == 0].dropna(how="any")
df_hate = df[df["Polarity"] == 1].dropna(how="any")

# df_nonhate has 1527 rows
# df_hate has 4797 rows
# downsample df_nonhate to 4797 rows
# cache the leftover data for testing purpose
df_nonhate, df_nonhate_leftover = (
    df_nonhate.head(len(df_hate)),
    df_nonhate.tail(len(df_nonhate) - len(df_hate)),
)
df_train_test = df_nonhate.append(df_hate).sample(frac=1)

# df_train_test has 9594 rows in total
# split it into train (80%) and test (20%) dataframes
X, y = df_train_test["Tweet"], df_train_test["Polarity"]
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
# train_X and train_y have 7675 rows
# test_X and test_y have 1919 rows

if __name__ == "__main__":
    train_X.to_csv(TRAIN_X_FILE)
    train_y.to_csv(TRAIN_Y_FILE)
    test_X.to_csv(TEST_X_FILE)
    test_y.to_csv(TEST_Y_FILE)
