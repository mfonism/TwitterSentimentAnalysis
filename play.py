import re

import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud

import utils


def ingest_train_filename(filename):
    # read csv
    data = pd.read_csv(filename)
    # filter out incomplete rows
    data = data[data.Sentiment.isnull() == False]
    data = data[data.SentimentText.isnull() == False]
    # convert the sentiment column data to integers
    data.Sentiment = data.Sentiment.map(int)
    # data was filtered, so rows were probaby
    # removed along with their respective indices
    # recalculate the indices, shifting rows upwards
    # so that there are no jumps in indices
    data.reset_index(inplace=True)
    data.drop("index", axis=1, inplace=True)
    return data


tokenizer = WordPunctTokenizer()


def data_cleaner(text):
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
        # tokenize,
        # then return a sentence of individual words strung together using spaces
        tokens = tokenizer.tokenize(temp)
        return (" ".join(tokens)).strip()
    except:
        return "NC"


tqdm.pandas(desc="progress-bar")


def post_process(data, n=1000000):
    data = data.head(n)
    data.SentimentText = data.SentimentText.progress_map(data_cleaner)
    data.reset_index(inplace=True)
    data.drop("index", inplace=True, axis=1)
    return data


def make_train(filename="dataset.csv"):
    train = ingest_train_filename("dataset.csv")
    train = post_process(train)


# style matplotlib for visualization
plt.style.use("fivethirtyeight")


def _visualize_neg(train):
    neg_tweets = train[train.Sentiment == 0]
    neg_string = []

    for t in neg_tweets.SentimentText:
        neg_string.append(t)

    neg_string = pd.Series(neg_string).str.cat(sep=" ")
    wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(
        neg_string
    )
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def _visualize_pos(train):
    pos_tweets = train[train.Sentiment == 1]
    pos_string = []

    for t in pos_tweets.SentimentText:
        pos_string.append(t)

    pos_string = pd.Series(pos_string).str.cat(sep=" ")
    wordcloud = WordCloud(
        width=1600, height=800, max_font_size=200, colormap="magma"
    ).generate(pos_string)
    # generate figure
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def save_to_csv(train):
    clean_data = pd.DataFrame(train, columns=["SentimentText"])
    clean_data["Sentiment"] = train.Sentiment

    clean_data.to_csv("clean_data.csv", encoding="utf-8")

    csv = "clean_data.csv"
    data = pd.read_csv(csv, index_col=0)
    data.head()


if __name__ == "__main__":
    # train = make_train()
    # save_to_csv(train)
    # _visualize_pos(train)
    # _visualize_neg(train)
    pass
