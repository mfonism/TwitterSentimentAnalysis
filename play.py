import re

from nltk.tokenize import WordPunctTokenizer
import pandas as pd

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
