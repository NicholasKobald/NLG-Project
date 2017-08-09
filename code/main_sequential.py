import numpy as np
import pandas as pd

import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, LSTM, Dense, Activation, Embedding, GRU, Dropout
from keras.models import Model, Sequential
from keras.utils.data_utils import Sequence

from sklearn.model_selection import train_test_split

from features import create_bow, load_word2vec, get_w2v_idx
from utils import generate_vocab, gen_or_load_feats

np.random.seed(7)

stance_id = {'agree': 0, 'discuss': 3, 'disagree': 1, 'unrelated': 2}


def create_dataset(name='train'):
    all_data = pd.read_csv('../data_sets/' + name + '_stances.csv')
    to_join = pd.read_csv('../data_sets/' + name + '_bodies.csv')
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


def stance_matrix(stances):
    stance_classes = np.fromiter((stance_id[s] for s in stances),
                                 np.int, len(stances)).reshape(-1, 1)
    return keras.utils.to_categorical(stance_classes, 4)

BATCH_SIZE = 64
# We may want to experiment with adjusting these two
MAX_HEAD_LEN = 20
MAX_ART_LEN = 200

def create_model(train_set, w2v, max_words=100):
    # parallel lists
    stances = train_set['Stance']
    articles = train_set['articleBody']
    headlines = train_set['Headline']

    print('Generating Word2vec features')
    # Train vocab and generate BOW representation
    train_arts = gen_or_load_feats(
        lambda: get_w2v_idx(articles, w2v, 200),
        '../features/w2v_articles{0}.npy'.format(200)
    )
    train_heads = gen_or_load_feats(
        lambda: get_w2v_idx(headlines, w2v, 20),
        '../features/w2v_headlines{0}.npy'.format(20)
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
    model.fit_generator(train_seq, len(train_heads) // BATCH_SIZE, epochs=2, verbose=1, 
                        use_multiprocessing=False, workers=8)
    return model


class WordVecSequence(Sequence):
    def __init__(self, w2v, heads, arts, stances=None, batch_size=64):
        self.w2v_mat = w2v.syn0
        self.heads = heads
        self.arts = arts
        self.stances = stances
        self.batch_size = batch_size
        self.num_entries = len(self.heads) // self.batch_size
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

MAX_WORDS = 5000

def main():
    train_dataset = create_dataset()
    train_set, eval_set = split_dataset(train_dataset)
    test_dataset = create_dataset(name='test')

    w2v = load_word2vec()
    print('Loaded word2vec')

    eval_Y = stance_matrix(eval_set['Stance'])
    eval_arts = get_w2v_idx(eval_set['articleBody'], w2v, 200)
    eval_heads = get_w2v_idx(eval_set['Headline'], w2v, 20)
    eval_seq = WordVecSequence(w2v, eval_heads, eval_arts, eval_Y, batch_size=BATCH_SIZE)
    
    model = create_model(train_set, w2v, MAX_WORDS)
    
    print('Evaluating model with {} entries ({} batches)'.format(
        len(eval_set), len(eval_set) // BATCH_SIZE))
    print(model.evaluate_generator(eval_seq, len(eval_set) // BATCH_SIZE, 
                              use_multiprocessing=False, workers=8))
    # print(model.predict_generator(eval_seq, len(eval_set) // BATCH_SIZE, 
    #                               use_multiprocessing=False, workers=8, verbose=1))
    exit(0)

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
