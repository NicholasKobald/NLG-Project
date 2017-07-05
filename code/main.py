import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model, Sequential
import keras

from utils import DataSet

np.random.seed(7)

stance_id = {
    'agree': 0,
    'discuss': 3,
    'disagree': 1,
    'unrelated': 2
}

def main():
    train_set = DataSet()
    test_set = DataSet('competition_test_bodies.csv',
                       'competition_test_stances.csv')

    # embedding_vector_length = 32
    max_words = 100

    tk = Tokenizer(num_words=max_words, lower=True, split=" ")

    # parallel lists
    stances = np.fromiter(map(lambda s: stance_id[s], train_set.triples['stances']),
                          np.int, len(train_set.triples['stances'])).reshape(-1, 1)
    articles = train_set.triples['articles']
    headlines = train_set.triples['headlines']

    # Train vocab and generate BOW representation
    tk.fit_on_texts(articles + headlines)
    bodies_bow = tk.texts_to_matrix(articles)
    heads_bow = tk.texts_to_matrix(headlines)

    train_seq = np.hstack((bodies_bow, heads_bow))
    print("Trained seq:", train_seq[:5])
    print("Seq has shape:", train_seq.shape)

    # Converts the N x 1 class vector to a N * classes binary matrix
    # This is needed to placate keras, for some bizarre reason
    train_stances = keras.utils.to_categorical(stances, 4)

    # model.add(Embedding(max_features, 100, input_length=max_len))
    # del.add(LSTM(100))
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
    # probably worth looking at:
    # https://keras.io/getting-started/sequential-model-guide/
    #
    #

    print("Create Sequential model")
    model = Sequential()
    # Input layer takes concatenated BOW vectors
    model.add(Dense(32, input_shape=(2 * max_words,)))
    model.add(Activation('relu'))
    # 4-class outputs - run argmax or on this to get a most probable class
    model.add(Dense(4))
    model.add(Activation('softmax'))
    # model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(train_seq, train_stances, batch_size=32, epochs=10, verbose=1, validation_split=0.1, shuffle=True)

    import textwrap
    test_idx = 0
    print('~~~ Test ~~~')
    print('Headline: ', headlines[test_idx])
    print('Article: ', textwrap.shorten(articles[test_idx], 1000))
    print('Stance: ', stances[test_idx])
    print('Predictions: ', model.predict(train_seq[0].reshape(1, -1), batch_size=1))

if __name__ == "__main__":
    main()
