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
df.dropna(how="any", inplace=True)

# df has 20071 rows in total (4794 for hate speech, 15247 for nonhate speech)
# split it into train (80%) and test (20%) dataframes
X, y = df["Tweet"], df["Polarity"]
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
# train_X and train_y have 16056 rows
# test_X and test_y have 4015 rows

if __name__ == "__main__":
    train_X.to_csv(TRAIN_X_FILE)
    train_y.to_csv(TRAIN_Y_FILE)
    test_X.to_csv(TEST_X_FILE)
    test_y.to_csv(TEST_Y_FILE)
