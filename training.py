from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
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


# do same for validation
sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=max_length)


# -------------------------------------
#
# -------------------------------------

model_cnn = Sequential()

e = Embedding(100000, 100, input_length=max_length)
model_cnn.add(e)
model_cnn.add(
    Conv1D(filters=100, kernel_size=2, padding="valid", activation="relu", strides=1)
)
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(256, activation="relu"))
model_cnn.add(Dense(1, activation="sigmoid"))
model_cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model_cnn.fit(
    x_train_seq,
    y_train,
    validation_data=(x_val_seq, y_validation),
    epochs=5,
    batch_size=32,
    verbose=2,
)
score, acc = model_cnn.evaluate(x_val_seq, y_validation, verbose=2, batch_size=32)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
