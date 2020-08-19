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


def create_class_dist(train):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0.170, 0.835])
    ax.set_xticklabels(["Not Hate Speech", "Hate Speech"])
    train.hist(bins=3, ax=ax, color="teal")
    plt.title("")
    plt.ylabel("Frequency")
    plt.xlabel("Class")
    plt.savefig(DATA_DIR / "distribution_of_classes.png", bbox_inches="tight")
    plt.close()


def create_wordcloud_neg(train):
    neg_tweets = train[train.Polarity == 1]
    neg_string = []

    for t in neg_tweets.Tweet:
        neg_string.append(t)

    neg_string = pd.Series(neg_string).str.cat(sep=" ")
    wordcloud = WordCloud(
        width=1600, height=800, max_font_size=200, colormap="magma"
    ).generate(neg_string)
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(DATA_DIR / "negative_cloud.png", bbox_inches="tight")
    plt.close()


def create_wordcloud_pos(train):
    pos_tweets = train[train.Polarity == 0]
    pos_string = []

    for t in pos_tweets.Tweet:
        pos_string.append(t)

    pos_string = pd.Series(pos_string).str.cat(sep=" ")
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(
        pos_string
    )
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(DATA_DIR / "positive_cloud.png", bbox_inches="tight")
    plt.close()


def create_most_frequent_bigrams_neg(train, n=20):
    fig, ax = plt.subplots(figsize=(12, 10))
    bigrams_series = pd.Series(_get_bigrams_neg(train)).value_counts()[:n]
    bigrams_series.sort_values().plot.barh(color="teal", width=0.9, ax=ax)
    # plt.title(f"{n} Most Frequently Occuring Bigrams (Hate-speech Tweets)")
    plt.ylabel("Bigram")
    plt.xlabel("Frequency")
    plt.savefig(DATA_DIR / "bigrams_neg.png", bbox_inches="tight")
    plt.close()


def create_most_frequent_bigrams_pos(train, n=20):
    fig, ax = plt.subplots(figsize=(12, 10))
    bigrams_series = pd.Series(_get_bigrams_pos(train)).value_counts()[:n]
    bigrams_series.sort_values().plot.barh(color="teal", width=0.9, ax=ax)
    # plt.title(f"{n} Most Frequently Occuring Bigrams (Non-hate-speech Tweets)")
    plt.ylabel("Bigram")
    plt.xlabel("Frequency")
    plt.savefig(DATA_DIR / "bigrams_pos.png", bbox_inches="tight")
    plt.close()


def create_most_frequent_trigrams_neg(train, n=20):
    fig, ax = plt.subplots(figsize=(12, 10))
    trigrams_series = pd.Series(_get_trigrams_neg(train)).value_counts()[:n]
    trigrams_series.sort_values().plot.barh(color="teal", width=0.9, ax=ax)
    # plt.title(f"{n} Most Frequently Occuring Trigrams (Hate-speech Tweets)")
    plt.ylabel("Trigram")
    plt.xlabel("Frequency")
    plt.savefig(DATA_DIR / "trigrams_neg.png", bbox_inches="tight")
    plt.close()


def create_most_frequent_trigrams_pos(train, n=20):
    fig, ax = plt.subplots(figsize=(12, 10))
    trigrams_series = pd.Series(_get_trigrams_pos(train)).value_counts()[:n]
    trigrams_series.sort_values().plot.barh(color="teal", width=0.9, ax=ax)
    # plt.title(f"{n} Most Frequently Occuring Trigrams (Non-hate-speech Tweets)")
    plt.ylabel("Trigram")
    plt.xlabel("Frequency")
    plt.savefig(DATA_DIR / "trigrams_pos.png", bbox_inches="tight")
    plt.close()


def _get_bigrams_neg(train):
    neg_tweets = train[train.Polarity == 1]
    list_of_neg_bigrams = neg_tweets.Tweet.map(
        lambda tweet: list(nltk.ngrams(tweet.split(), 2))
    ).to_list()
    return [item for sublist in list_of_neg_bigrams for item in sublist]


def _get_trigrams_neg(train):
    neg_tweets = train[train.Polarity == 1]
    list_of_neg_trigrams = neg_tweets.Tweet.map(
        lambda tweet: list(nltk.ngrams(tweet.split(), 3))
    ).to_list()
    return [item for sublist in list_of_neg_trigrams for item in sublist]


def _get_bigrams_pos(train):
    pos_tweets = train[train.Polarity == 0]
    list_of_pos_bigrams = pos_tweets.Tweet.map(
        lambda tweet: list(nltk.ngrams(tweet.split(), 2))
    ).to_list()
    return [item for sublist in list_of_pos_bigrams for item in sublist]


def _get_trigrams_pos(train):
    pos_tweets = train[train.Polarity == 0]
    list_of_pos_trigrams = pos_tweets.Tweet.map(
        lambda tweet: list(nltk.ngrams(tweet.split(), 3))
    ).to_list()
    return [item for sublist in list_of_pos_trigrams for item in sublist]


if __name__ == "__main__":
    # style matplotlib for visualization
    plt.style.use("fivethirtyeight")

    # process data
    print()
    print("Processing data...")
    df = get_dataframe()
    df = process_dataframe(df)
    save_dataframe(df)
    print("Done!")

    # create class distribution
    print()
    print("Creating class distribution...")
    create_class_dist(df)

    # create word clouds
    print()
    print("Creating word cloud...")
    create_wordcloud_pos(df)
    create_wordcloud_neg(df)

    # create bigrams
    print()
    print("Creating bigrams...")
    create_most_frequent_bigrams_pos(df)
    create_most_frequent_bigrams_neg(df)

    # create trigrams
    print()
    print("Creating trigrams")
    create_most_frequent_trigrams_pos(df)
    create_most_frequent_trigrams_neg(df)
