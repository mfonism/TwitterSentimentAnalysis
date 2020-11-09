from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..vectorization.data_splitting import train_X, train_y, test_X, test_y
from ..utils.math import round_up_to_nearest_ten

MAX_LEN = round_up_to_nearest_ten(max((len(row.split()) for row in train_X)))

vectorizer = TfidfVectorizer(ngram_range=(1, 4)).fit(train_X)
transformed_train_X = vectorizer.transform(train_X)
transformed_test_X = vectorizer.transform(test_X)

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(train_X)

train_X_seq = tokenizer.texts_to_sequences(train_X)
train_X_seq = pad_sequences(train_X_seq, maxlen=MAX_LEN)

test_X_seq = tokenizer.texts_to_sequences(test_X)
test_X_seq = pad_sequences(test_X_seq, maxlen=MAX_LEN)

input_tfidf = Input(shape=(300,))
input_text = Input(shape=(MAX_LEN,))

# embedding layer
embedding = Embedding(100000, 100, input_length=MAX_LEN)(input_text)

# 1D convolution layer
convolution = Conv1D(
    filters=100, kernel_size=2, padding="valid", activation="relu", strides=1
)(embedding)

# 1D max pooling layer
max_pooling = GlobalMaxPooling1D()(convolution)

# dense layers
dense1 = Dense(256, activation="relu")(convolution)
dense2 = Dense(1, activation="sigmoid")(dense1)

# model
model_cnn = Model(input_text, dense2)
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
    json_file = MODELS_DIR / "model_cnn_with_tfidf_untrained.json"
    json_file.write_text(model_cnn.to_json())

    # save the trained/weighted model to H5
    h5_file = MODELS_DIR / "model_cnn_with_tfidf_trained.h5"
    model_cnn.save(h5_file, save_format="h5")

    score, accuracy = model_cnn.evaluate(test_X_seq, test_y, verbose=2, batch_size=32)

    print(f"Score:    {score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    run()
