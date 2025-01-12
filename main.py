import numpy as np
import pandas as pd

import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Activation, Embedding, GRU, Dropout
from keras.models import Model, Sequential

from sklearn.model_selection import train_test_split

from features import create_bow, load_word2vec
from utils import generate_vocab

np.random.seed(7)

stance_id = {'agree': 0, 'discuss': 3, 'disagree': 1, 'unrelated': 2}

MAX_WORDS = 5000



def stance_matrix(stances):
    stance_classes = np.fromiter((stance_id[s] for s in stances),
                                 np.int, len(stances)).reshape(-1, 1)
    return keras.utils.to_categorical(stance_classes, 4)


def create_model(train_set, vocab, max_words=100):
    # parallel lists
    stances = train_set['Stance']
    articles = train_set['articleBody']
    headlines = train_set['Headline']

    # Train vocab and generate BOW representation
    train_seq = create_bow(articles, headlines, vocab)
    print("Trained seq:", train_seq[:5])
    print("Seq has shape:", train_seq.shape)

    # Converts the N x 1 class vector to a N * classes binary matrix
    # This is needed to placate keras, for some bizarre reason
    train_stances = stance_matrix(stances)

    print("Create Sequential model")
    model = Sequential()

    # Input layer takes concatenated BOW vectors
    model.add(Dense(600, input_shape=(2 * max_words,), activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Embedding(2, 200, input_length=2 * max_words))
    # model.add(GRU(10))
    # 4-class outputs - run argmax or on this to get a most probable class
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_seq, train_stances, batch_size=64, epochs=3, verbose=1, validation_split=0.1, shuffle=True)
    return model


def split_dataset(dataset):
    train_set, test_set = train_test_split(dataset, stratify=dataset['Stance'])
    # train_set = even_classes(train_set, sample=2000)
    return train_set, test_set



def main():
    train_dataset = create_dataset()
    train_set, eval_set = split_dataset(train_dataset)
    test_dataset = create_dataset(name='test')

    vocab = generate_vocab(pd.concat([train_set, test_dataset]), MAX_WORDS, stop_words='english')
    # tk = Tokenizer(num_words=MAX_WORDS, lower=True, split=" ")
    model = create_model(train_set, vocab, MAX_WORDS)

    eval_X = create_bow(eval_set['articleBody'], eval_set['Headline'], vocab)
    eval_Y = stance_matrix(eval_set['Stance'])
    print(model.evaluate(eval_X, eval_Y, verbose=1))

    test_X = create_bow(test_dataset['articleBody'], test_dataset['Headline'], vocab)
    test_Y = stance_matrix(test_dataset['Stance'])

    print(model.evaluate(test_X, test_Y, verbose=1))

    # import textwrap
    # test_idx = train_set[train_set['Stance'] == 'unrelated'].index[0]
    # print('~~~ Test ~~~')
    # print('Headline: ', headlines[test_idx])
    # print('Article: ', textwrap.shorten(articles[test_idx], 1000))
    # print('Stance: ', stances[test_idx])
    # pred = model.predict(train_seq[test_idx].reshape(1, -1), batch_size=1)[0]
    # print('Predictions: ', {k: pred[stance_id[k]] for k in stance_id})
    # print('Top prediction: ', np.argmax(pred))

if __name__ == "__main__":
    main()
