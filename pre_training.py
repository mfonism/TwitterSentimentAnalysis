import csv

import utils
from config import DATA_DIR


def collate_parts():
    PARTS_DIR = DATA_DIR / "parts"

    with open(DATA_DIR / "stitched.csv", "wt", newline="") as writefile:

        count = 0
        writer = csv.DictWriter(writefile, fieldnames=["Tweet", "Polarity"])
        writer.writeheader()

        for child in PARTS_DIR.iterdir():
            # skip nested directories
            # and non-csv files
            if not child.is_file() or child.suffix != ".csv":
                print(f"Skipping: {child!s}")
                continue
            with child.open(newline="") as readfile:
                reader = csv.DictReader(readfile)
                for row in reader:
                    # skip rows with empty values in any column
                    if not all(row.values()):
                        continue
                    writer.writerow(
                        {
                            "Tweet": utils.clean_string(row["b'message'"]),
                            "Polarity": utils.clean_string(row["b'polarity'"]),
                        }
                    )
                    count += 1

        print(f"Written {count} rows.")


if __name__ == "__main__":
    collate_parts()
