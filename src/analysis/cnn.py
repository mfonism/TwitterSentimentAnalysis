from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
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

model_cnn = Sequential()

model_cnn.add(Embedding(100000, 100, input_length=MAX_LEN))
model_cnn.add(
    Conv1D(filters=100, kernel_size=2, padding="valid", activation="relu", strides=1)
)
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dense(256, activation="relu"))
model_cnn.add(Dense(1, activation="sigmoid"))
model_cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


def run():
    from ..config import MODELS_DIR

    model_cnn.fit(
        train_X_seq,
        train_y,
        validation_data=(test_X_seq, test_y),
        epochs=5,
        batch_size=32,
        verbose=2,
    )

    # save the untrained model to JSON
    json_file = MODELS_DIR / "model_cnn_untrained.json"
    json_file.write_text(model_cnn.to_json())

    # save the trained/weighted model to H5
    h5_file = MODELS_DIR / "model_cnn_trained.h5"
    model_cnn.save(h5_file, save_format="h5")

    score, accuracy = model_cnn.evaluate(test_X_seq, test_y, verbose=2, batch_size=32)

    print(f"Score:    {score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    run()
