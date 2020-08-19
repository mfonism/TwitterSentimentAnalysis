import csv
import re

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from wordcloud import WordCloud

from config import CLEAN_DATA_FILE, DATA_DIR, PARTS_DIR, STITCHED_FILE
from utils import text_utils


def collate_parts():
    with open(STITCHED_FILE, "wt", newline="") as writefile:

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
                            "Tweet": text_utils.strip_byte(row["b'message'"]),
                            "Polarity": text_utils.strip_byte(row["b'polarity'"]),
                        }
                    )
                    count += 1

        print(f"Written {count} rows.")


def remove_stopwords(tweet, is_clean=False):
    if not is_clean:
        tweet = text_utils.clean_tweet(tweet)
    tokens = WordPunctTokenizer().tokenize(tweet)
    english_stopwords = stopwords.words("english")
    return " ".join(token for token in tokens if not token in english_stopwords).strip()


def get_dataframe(csvfile=STITCHED_FILE):
    """Return dataframe created from input csv file."""
    return pd.read_csv(csvfile)


def process_dataframe(df):
    df.dropna(how="any", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Tweet"] = df["Tweet"].map(remove_stopwords)
    df["Polarity"] = df["Polarity"].map(int)
    return df


def save_dataframe(df, csvfile=CLEAN_DATA_FILE):
    """Save dataframe to input csv file."""
    df.to_csv(csvfile)


def create_class_dist(df):
    df_polarity = df["Polarity"]

    fig = plt.figure(figsize=(16, 10))

    ax = fig.gca()
    ax.set_xlim([-0.1, 3])
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["Neutral Speech", "Hate Speech", "Offensive Speech"])
    ax.hist(df_polarity, bins=[0, 1.0, 2.0, 2.9], width=0.9, color="green", alpha=0.5)

    plt.savefig(DATA_DIR / "distribution_of_classes.png", bbox_inches="tight")
    plt.close()


def create_wordcloud(df, outfile, colormap=None):
    concatenated_tweets = df["Tweet"].str.cat(sep=" ")

    wordcloud = WordCloud(
        width=1600, height=1000, max_font_size=256, colormap=colormap
    ).generate(concatenated_tweets)

    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def create_modal_bigrams(df, outfile, n=20):
    fig, ax = plt.subplots(figsize=(16, 10))
    bigram_series = pd.Series(_get_bigrams(df)).value_counts()[:n]
    bigram_series.sort_values().plot.barh(color="green", alpha=0.5, width=1, ax=ax)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def create_modal_trigrams(df, outfile, n=20):
    fig, ax = plt.subplots(figsize=(16, 10))
    bigram_series = pd.Series(_get_trigrams(df)).value_counts()[:n]
    bigram_series.sort_values().plot.barh(color="green", alpha=0.5, width=1, ax=ax)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def _get_bigrams(df):
    list_of_bigrams = (
        df["Tweet"].map(lambda tweet: list(nltk.ngrams(tweet.split(), 2))).to_list()
    )
    return [item for sublist in list_of_bigrams for item in sublist]


def _get_trigrams(df):
    list_of_trigrams = (
        df["Tweet"].map(lambda tweet: list(nltk.ngrams(tweet.split(), 3))).to_list()
    )
    return [item for sublist in list_of_trigrams for item in sublist]


if __name__ == "__main__":
    # style matplotlib for visualization
    plt.style.use("fivethirtyeight")

    # process data
    print()
    print("Processing data...")
    df = get_dataframe()
    df = process_dataframe(df)
    save_dataframe(df)

    # categorize data
    print()
    print(
        "Categorizing data according to polarity -- 0 == neutral, 1 == hate, 2 == offensive..."
    )
    df_neutral = df[df["Polarity"] == 0]
    df_hate = df[df["Polarity"] == 1]
    df_offensive = df[df["Polarity"] == 2]

    print(f"Num neut: {df_neutral.shape[0]:>5}")
    print(f"Num hate: {df_hate.shape[0]:>5}")
    print(f"Num offe: {df_offensive.shape[0]:>5}")

    # create class distribution
    print()
    print("Creating class distribution...")
    create_class_dist(df)

    # create word clouds
    print()
    print("Creating neutral word cloud...")
    create_wordcloud(
        df_neutral, outfile=(DATA_DIR / "wordcloud--neutral.png"), colormap="Purples"
    )
    print("Creating hate word cloud...")
    create_wordcloud(
        df_hate, outfile=(DATA_DIR / "wordcloud--hate.png"), colormap="Reds_r"
    )
    print("Creating offensive word cloud...")
    create_wordcloud(
        df_offensive,
        outfile=(DATA_DIR / "wordcloud--offensive.png"),
        colormap="OrRd_r",
    )

    # create bigrams
    print()

    print("Creating neutral bigrams...")
    create_modal_bigrams(df_neutral, outfile=(DATA_DIR / "bigrams--neutral.png"))

    print("Creating hate bigrams...")
    create_modal_bigrams(df_hate, outfile=(DATA_DIR / "bigrams--hate.png"))

    print("Creating offensive bigrams...")
    create_modal_bigrams(df_offensive, outfile=(DATA_DIR / "bigrams--offensive.png"))

    # create trigrams
    print()
    print("Creating neutral trigrams...")
    create_modal_trigrams(df_neutral, outfile=(DATA_DIR / "trigrams--neutral.png"))

    print("Creating hate trigrams...")
    create_modal_trigrams(df_hate, outfile=(DATA_DIR / "trigrams--hate.png"))

    print("Creating offensive trigrams...")
    create_modal_trigrams(df_offensive, outfile=(DATA_DIR / "trigrams--offensive.png"))

    print()
    print("DONE!")
