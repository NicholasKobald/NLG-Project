import numpy as np
import pandas as pd

from functools import reduce

from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model, Sequential
import keras

from sklearn.model_selection import train_test_split

from utils import DataSet

np.random.seed(7)

stance_id = {
    'agree': 0,
    'discuss': 3,
    'disagree': 1,
    'unrelated': 2
}


def create_dataset():
    all_data = pd.read_csv('../data_sets/train_stances.csv')
    to_join = pd.read_csv('../data_sets/train_bodies.csv')
    return pd.merge(all_data, to_join)


def even_classes(data, sample='min_class'):
    sample_n = sample

    groups = data.groupby('Stance')
    counts = groups.size()
    if sample == 'min_class':
        sample_n = min(counts)
    elif sample == 'max_class':
        sample_n = max(counts)

    sampled = map(lambda g: g[1].sample(sample_n, replace=True), groups)
    return pd.concat(sampled).reset_index(drop=True)


def main():
    print("Loading training set")
    train_set = DataSet()
    test_set = DataSet('competition_test_bodies.csv',
                       'competition_test_stances.csv')

    max_words = 100
    tk = Tokenizer(num_words=max_words, lower=True, split=" ")

    stances = np.fromiter(map(lambda s: stance_id[s], train_set.triples['stances']),
                          np.int, len(train_set.triples['stances'])).reshape(-1, 1)
    #for evaluation
    test_stances = np.fromiter(map(lambda s: stance_id[s], test_set.triples['stances']),
                           np.int, len(test_set.triples['stances'])).reshape(-1, 1)


    articles = train_set.triples['articles']
    headlines = train_set.triples['headlines']

    test_articles = test_set.triples['articles']
    test_headlines = test_set.triples['headlines']

    assert(len(test_articles) == len(test_headlines))
    print("Testing on {} articles".format(len(test_articles)))



    # Train vocab and generate BOW representation
    print("Fitting Tokenizer on texts")
    tk.fit_on_texts(articles + headlines)
    bodies_bow = tk.texts_to_matrix(articles)
    heads_bow = tk.texts_to_matrix(headlines)

    bodies_bow_test = tk.texts_to_matrix(test_articles)
    heads_bow_test = tk.texts_to_matrix(test_headlines)

    test_seq  = np.hstack((bodies_bow_test, heads_bow_test))
    train_seq = np.hstack((bodies_bow, heads_bow))


    # Converts the N x 1 class vector to a N * classes binary matrix
    # This is needed to placate keras, for some bizarre reason
    stance_classes = np.fromiter((stance_id[s] for s in stances),
                                 np.int, len(stances)).reshape(-1, 1)
    train_stances = keras.utils.to_categorical(stance_classes, 4)


    print("Create Sequential model")
    model = Sequential()

    # Input layer takes concatenated BOW vectors
    model.add(Dense(32, input_shape=(2 * max_words,)))
    model.add(Activation('relu'))
    # 4-class outputs - run argmax or on this to get a most probable class
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(train_seq, train_stances, batch_size=32, epochs=10, verbose=1, validation_split=0.1, shuffle=True)

    import textwrap
    test_idx = train_set[train_set['Stance'] == 'unrelated'].index[0]
    print('~~~ Test ~~~')
    print('Headline: ', headlines[test_idx])
    print('Article: ', textwrap.shorten(articles[test_idx], 1000))
    print('Stance: ', stances[test_idx])
    print('Predictions: ', model.predict(test_seq.reshape(1, -1), batch_size=1))

if __name__ == "__main__":
    main()
