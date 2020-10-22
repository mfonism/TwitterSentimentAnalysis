import pathlib


BASE_DIR = pathlib.Path(__file__).absolute().parent.parent
DATA_DIR = BASE_DIR / "data"
PARTS_DIR = DATA_DIR / "parts"
TRAIN_AND_TEST_DATA_DIR = DATA_DIR / "train_and_test"

STITCHED_FILE = DATA_DIR / "stitched.csv"
CLEAN_DATA_FILE = DATA_DIR / "clean_data.csv"

TRAIN_X_FILE = TRAIN_AND_TEST_DATA_DIR / "train_X.csv"
TEST_X_FILE = TRAIN_AND_TEST_DATA_DIR / "test_X.csv"
TRAIN_Y_FILE = TRAIN_AND_TEST_DATA_DIR / "train_y.csv"
TEST_Y_FILE = TRAIN_AND_TEST_DATA_DIR / "test_y.csv"

# some sort of seed for randomization
RANDOM_STATE = 69
