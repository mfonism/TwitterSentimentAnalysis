from keras.layers import Dense, LSTM, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from ..vectorization.data_splitting import train_X, train_y, test_X, test_y
from ..utils.math import round_up_to_nearest_ten

MAX_LEN = round_up_to_nearest_ten(max((len(row.split()) for row in train_X)))
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(train_X)

train_X_seq = tokenizer.texts_to_sequences(train_X)
train_X_seq = pad_sequences(train_X_seq, maxlen=MAX_LEN)

test_X_seq = tokenizer.texts_to_sequences(test_X)
test_X_seq = pad_sequences(test_X_seq, maxlen=MAX_LEN)

model_lstm = Sequential()

model_lstm.add(Embedding(100000, 128))
model_lstm.add(SpatialDropout1D(0.4))
model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(256, activation="relu"))
model_lstm.add(Dense(1, activation="sigmoid"))
model_lstm.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


def run():
    from ..config import MODELS_DIR

    model_lstm.fit(
        train_X_seq, train_y, epochs=10, batch_size=32, verbose=2,
    )

    # save the untrained model to JSON
    json_file = MODELS_DIR / "model_lstm_untrained.json"
    json_file.write_text(model_lstm.to_json())

    # save the trained/weighted model to H5
    h5_file = MODELS_DIR / "model_lstm_trained.h5"
    model_lstm.save(h5_file, save_format="h5")

    score, accuracy = model_lstm.evaluate(test_X_seq, test_y, verbose=2, batch_size=32)

    print(f"Score:    {score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    run()
