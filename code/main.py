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
    dataset = create_dataset()
    train_set, test_set = train_test_split(dataset, stratify=dataset['Stance'])
    train_set = even_classes(train_set, sample=2000)
    print('train set:', train_set)
    # test_set = DataSet('competition_test_bodies.csv',
    #                    'competition_test_stances.csv')

    # embedding_vector_length = 32
    max_words = 100

    tk = Tokenizer(num_words=max_words, lower=True, split=" ")

    # parallel lists
    stances = train_set['Stance']
    articles = train_set['articleBody']
    headlines = train_set['Headline']

    # Train vocab and generate BOW representation
    tk.fit_on_texts(articles.append(headlines))
    bodies_bow = tk.texts_to_matrix(articles)
    heads_bow = tk.texts_to_matrix(headlines)

    train_seq = np.hstack((bodies_bow, heads_bow))
    print("Trained seq:", train_seq[:5])
    print("Seq has shape:", train_seq.shape)

    # Converts the N x 1 class vector to a N * classes binary matrix
    # This is needed to placate keras, for some bizarre reason
    stance_classes = np.fromiter((stance_id[s] for s in stances),
                                 np.int, len(stances)).reshape(-1, 1)
    train_stances = keras.utils.to_categorical(stance_classes, 4)

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
    test_idx = train_set[train_set['Stance'] == 'unrelated'].index[0]
    print('~~~ Test ~~~')
    print('Headline: ', headlines[test_idx])
    print('Article: ', textwrap.shorten(articles[test_idx], 1000))
    print('Stance: ', stances[test_idx])
    pred = model.predict(train_seq[test_idx].reshape(1, -1), batch_size=1)[0]
    print('Predictions: ', {k: pred[stance_id[k]] for k in stance_id})
    print('Top prediction: ', np.argmax(pred))

if __name__ == "__main__":
    main()
