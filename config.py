import pathlib


BASE_DIR = pathlib.Path(__file__).absolute().parent
DATA_DIR = BASE_DIR / "data"
PARTS_DIR = DATA_DIR / "parts"

STITCHED_FILE = DATA_DIR / "stitched.csv"
CLEAN_DATA_FILE = DATA_DIR / "clean_data.csv"
