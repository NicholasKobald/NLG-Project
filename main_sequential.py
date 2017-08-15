import numpy as np
import pandas as pd

import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Activation, Embedding, GRU, Dropout
from keras.models import Model, Sequential
from keras.utils.data_utils import Sequence

from sklearn.model_selection import train_test_split

from features import create_bow, get_w2v_idx
from utils import generate_vocab, gen_or_load_feats, load_word2vec
from utils import create_dataset, even_classes

from score import report_score, LABELS

from collections import Counter

np.random.seed(7)

stance_id = {'agree': 0, 'discuss': 2, 'disagree': 1, 'unrelated': 3}

MAX_WORDS = 5000

BATCH_SIZE = 64
MAX_HEAD_LEN = 20
MAX_ART_LEN = 200



def stance_matrix(stances):
    stance_classes = np.fromiter((stance_id[s] for s in stances),
                                 np.int, len(stances)).reshape(-1, 1)
    return keras.utils.to_categorical(stance_classes, 4)



def create_model(train_set, w2v, max_words=100):
    # parallel lists
    stances = train_set['Stance']
    articles = train_set['articleBody']
    headlines = train_set['Headline']

    print('Generating Word2vec features')
    # Train vocab and generate BOW representation
    train_arts = gen_or_load_feats(
        lambda: get_w2v_idx(articles, w2v, 200),
        'features/w2v_articles{0}.npy'.format(200)
    )
    train_heads = gen_or_load_feats(
        lambda: get_w2v_idx(headlines, w2v, 20),
        'features/w2v_headlines{0}.npy'.format(20)
    )

    print('Initialize layers')
    input_arts = Input(shape=(MAX_ART_LEN, 300))
    input_heads = Input(shape=(MAX_HEAD_LEN, 300))

    # TODO play with increasing the cell counts
    lstm_arts = LSTM(20)(input_arts)
    lstm_heads = LSTM(20)(input_heads)

    merged = keras.layers.concatenate([lstm_heads, lstm_arts])
    linear = Dense(4, activation='tanh')(merged)
    predictions = Activation('softmax')(linear)

    # Converts the N x 1 class vector to a N * classes binary matrix
    # This is needed to placate keras, for some bizarre reason
    train_stances = stance_matrix(stances)

    print("Create Sequential model")
    model = Model(inputs=[input_heads, input_arts], outputs=[predictions])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_seq = WordVecSequence(w2v, train_heads, train_arts, train_stances)
    model.fit_generator(train_seq, round(len(train_heads) / BATCH_SIZE),
                        epochs=1, verbose=1, use_multiprocessing=False, workers=8)
    return model


class WordVecSequence(Sequence):

    def __init__(self, w2v, heads, arts, stances=None, batch_size=64):
        self.w2v_mat = w2v.syn0
        self.heads = heads
        self.arts = arts
        self.stances = stances
        self.batch_size = batch_size
        self.num_entries = round(len(self.heads) / self.batch_size)
        print(len(self.heads), self.num_entries)

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        start_idx = self.batch_size * idx
        end_idx = start_idx + self.batch_size
        batch_X = [
            self.w2v_mat[self.heads[start_idx:end_idx]],
            self.w2v_mat[self.arts[start_idx:end_idx]]
        ]
        if self.stances is None:
            return batch_X
        batch_Y = self.stances[start_idx:end_idx]
        return [batch_X, batch_Y]


def split_dataset(dataset):
    train_set, test_set = train_test_split(dataset, stratify=dataset['Stance'])
    # train_set = even_classes(train_set, sample=2000)
    return train_set, test_set


def main():
    train_dataset = create_dataset()
    train_set, eval_set = split_dataset(train_dataset)
    test_dataset = create_dataset(name='test')

    w2v = load_word2vec(
        fname='data_sets/word2vec_obj.object',
        bin_fname='data_sets/GoogleNews-vectors-negative300.bin'
    )
    print('Loaded word2vec')

    # eval_Y = stance_matrix(eval_set['Stance'])
    # eval_arts = get_w2v_idx(eval_set['articleBody'], w2v, 200)
    # eval_heads = get_w2v_idx(eval_set['Headline'], w2v, 20)
    # eval_seq = WordVecSequence(w2v, eval_heads, eval_arts, eval_Y, batch_size=BATCH_SIZE)

    model = create_model(train_set, w2v, MAX_WORDS)

    # print('Evaluating model with {} entries ({} batches)'.format(
    #     len(eval_set), len(eval_set) // BATCH_SIZE))
    # print(model.evaluate_generator(eval_seq, len(eval_set) // BATCH_SIZE,
    #                           use_multiprocessing=False, workers=8))
    # print(model.predict_generator(eval_seq, len(eval_set) // BATCH_SIZE,
    #                               use_multiprocessing=False, workers=8, verbose=1))

    test_Y = stance_matrix(test_dataset['Stance'])
    test_arts = get_w2v_idx(test_dataset['articleBody'], w2v, 200)
    test_heads = get_w2v_idx(test_dataset['Headline'], w2v, 20)
    # test_seq = WordVecSequence(w2v, test_heads, test_arts, test_Y, batch_size=BATCH_SIZE)
    test_seq = WordVecSequence(w2v, test_heads, test_arts, batch_size=BATCH_SIZE)

    print('Evaluating model with {} entries ({} batches)'.format(len(test_dataset), len(test_dataset) // BATCH_SIZE))
    # print(model.evaluate_generator(test_seq, len(test_dataset) // BATCH_SIZE, use_multiprocessing=False, workers=8))
    preds = model.predict_generator(test_seq, round(len(test_dataset) / BATCH_SIZE),
                                    use_multiprocessing=False, workers=8, verbose=1)
    preds = preds.argmax(axis=1)
    # print(preds.argmax(axis=1).shape)
    print(Counter(preds))
    pred_stances = [LABELS[p] for p in preds]
    print(Counter(pred_stances))
    report_score(list(test_dataset['Stance']), pred_stances)


if __name__ == "__main__":
    main()
