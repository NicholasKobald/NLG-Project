import numpy as np

import keras.preprocessing.text
from keras.preprocessing import sequence
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding

from utils import DataSet

np.random.seed(7)

stance_id = dict(
        agree=1,
        discuss=4,
        disagree=2,
        unrelated=3,
)

def main():
    train_set = DataSet()
    test_set = DataSet('competition_test_bodies.csv',
                       'competition_test_stances.csv')


    #embedding_vector_length = 32
    max_features = 2000
    max_len = 100

    tk = keras.preprocessing.text.Tokenizer(
        num_words=2000,
        lower=True,
        split=" "
    )

    #parrallel lists
    stance_to_num = lambda s: stance_id[s]
    stances = np.asarray(map(stance_to_num, train_set.triples['stances']))
    articles = np.asarray(train_set.triples['articles'])
    headlines = np.asarray(train_set.triples['headlines'])

    tk.fit_on_texts(articles)
    trained_seq = tk.texts_to_sequences(articles)
    trained_seq = sequence.pad_sequences(trained_seq, max_len)

    print("Trained seq:", trained_seq[:5])
    #print("Trained seq:", trained_seq[])
    print("Seq has shape:", trained_seq.shape)
    model = Sequential()
    print("Create Sequential model")
    #model.add(Embedding(max_features, 100, input_length=max_len))
    #del.add(LSTM(100))
    # NOTE
    # Trained seq is an embedding vector of the words. it has the shape
    # (49972, 100) since we padded the sequences to the max length.
    #
    # Attempting to add one layer to the neural net shouldn't be too bad.
    # The input is a numpy array of length 49972 (one for each article)
    # where each entry is an array of length 100.
    # however I get errors when I try to set
    # the input shape to he first layer as (None, 100) (none should be any pos integer)
    #
    # https://keras.io/layers/core/
    #
    #probably worth looking at:
    # https://keras.io/getting-started/sequential-model-guide/
    #
    #

    model.add(Dense(1, input_shape=(None, 100)))
    #model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(trained_seq, stances,
        batch_size=32, epochs=10,
        verbose=1, callbacks=None,
        validation_split=0.0, validation_data=None,
        shuffle=True, class_weight=None,
        sample_weight=None, initial_epoch=0
    )

if __name__ == "__main__":
    main()
