import csv
import re

import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud

import utils
from config import CLEAN_DATA_FILE, DATA_DIR, PARTS_DIR, STITCHED_FILE


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
                            "Tweet": utils.clean_string(row["b'message'"]),
                            "Polarity": utils.clean_string(row["b'polarity'"]),
                        }
                    )
                    count += 1

        print(f"Written {count} rows.")


def ingest_collated():
    # read csv
    data = pd.read_csv(STITCHED_FILE)
    # filter out incomplete rows
    data = data[data.Tweet.isnull() == False]
    data = data[data.Polarity.isnull() == False]
    # convert the polarity column data to integers
    data.Polarity = data.Polarity.map(int)
    # data was filtered, so rows were probaby
    # removed along with their respective indices
    # recalculate the indices, shifting rows upwards
    # so that there are no jumps in indices
    data.reset_index(inplace=True)
    data.drop("index", axis=1, inplace=True)
    return data


def data_cleaner(text):
    tokenizer = WordPunctTokenizer()
    try:
        # remove URLs
        temp = re.sub(utils.combined_pat, "", text)
        temp = re.sub(utils.www_pat, "", temp)
        # remove HTML tags
        temp = re.sub(utils.html_tag_pat, "", temp)
        # change to lowercase and
        # replace contracted negations with longer form
        temp = temp.lower()
        temp = utils.negations_pat.sub(lambda x: utils.negations_[x.group()], temp)
        # keep only the letters
        temp = re.sub("[^a-zA-Z]", " ", temp)
        # remove randos
        temp = re.sub(utils.rando_pat, "", temp)
        # tokenize,
        # then return a sentence of individual words strung together using spaces
        tokens = tokenizer.tokenize(temp)
        return (" ".join(tokens)).strip()
    except:
        return "NC"


tqdm.pandas(desc="progress-bar")


def post_process(data, n=1000000):
    data = data.head(n)
    data.Tweet = data.Tweet.progress_map(data_cleaner)
    data.reset_index(inplace=True)
    data.drop("index", inplace=True, axis=1)
    return data


def make_train():
    train = ingest_collated()
    train = post_process(train)
    return train


def save_to_csv(train):
    clean_data = pd.DataFrame(train, columns=["Tweet"])
    clean_data["Polarity"] = train.Polarity

    clean_data.to_csv(CLEAN_DATA_FILE, encoding="utf-8")

    data = pd.read_csv(CLEAN_DATA_FILE, index_col=0)
    data.head()


# style matplotlib for visualization
plt.style.use("fivethirtyeight")


def create_wordcloud_neg(train):
    neg_tweets = train[train.Polarity == 0]
    neg_string = []

    for t in neg_tweets.Tweet:
        neg_string.append(t)

    neg_string = pd.Series(neg_string).str.cat(sep=" ")
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(
        neg_string
    )
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(DATA_DIR / "negative_cloud.png", bbox_inches="tight")
    plt.close()


def create_wordcloud_pos(train):
    pos_tweets = train[train.Polarity == 1]
    pos_string = []

    for t in pos_tweets.Tweet:
        pos_string.append(t)

    pos_string = pd.Series(pos_string).str.cat(sep=" ")
    wordcloud = WordCloud(
        width=1600, height=800, max_font_size=200, colormap="magma"
    ).generate(pos_string)
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(DATA_DIR / "positive_cloud.png", bbox_inches="tight")
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


def create_class_dist(train):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([0.170, 0.835])
    ax.set_xticklabels(["Not Hate Speech", "Hate Speech"])
    train.hist(bins=3, ax=ax, color="teal")
    plt.title("Distribution of Classes")
    plt.savefig(DATA_DIR / "distribution_of_classes.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    collate_parts()
    train = make_train()
    save_to_csv(train)
