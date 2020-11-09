import pathlib


BASE_DIR = pathlib.Path(__file__).absolute().parent.parent
BASE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR = DATA_DIR / "parts"
PARTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_AND_TEST_DATA_DIR = DATA_DIR / "train_and_test"
TRAIN_AND_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STITCHED_FILE = DATA_DIR / "stitched.csv"
CLEAN_DATA_FILE = DATA_DIR / "clean_data.csv"

TRAIN_X_FILE = TRAIN_AND_TEST_DATA_DIR / "train_X.csv"
TEST_X_FILE = TRAIN_AND_TEST_DATA_DIR / "test_X.csv"
TRAIN_Y_FILE = TRAIN_AND_TEST_DATA_DIR / "train_y.csv"
TEST_Y_FILE = TRAIN_AND_TEST_DATA_DIR / "test_y.csv"

# some sort of seed for randomization
RANDOM_STATE = 69
