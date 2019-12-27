import pandas as pd


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
