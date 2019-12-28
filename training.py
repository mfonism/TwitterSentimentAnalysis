from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 2000


def load_clean_data():
    return pd.read_csv("clean_data.csv", index_col=0, encoding="latin-1")


def split_clean_data():
    """
    Split clean data into training and validation set.

    Returns:
    x_train, x_validation, y_train, y_validation
    """
    data = load_clean_data()
    return train_test_split(
        data.SentimentText, data.Sentiment, test_size=0.2, random_state=SEED
    )


x_train, x_validation, y_train, y_validation = split_clean_data()

# use Keras Tokenizer to split each word in a sentence
# then in order to get a sequential representation of each row
# use texts_to_sequences method to represent each word
# by a number
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

# check for the max length of rows in the corpus
# for padding
max_length = max(len(x.split()) for x in x_train)
# pad the sequence with the max length
x_train_seq = pad_sequences(sequences, maxlen=max_length)
