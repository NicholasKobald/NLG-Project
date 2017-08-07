import warnings #couldn't find a way around this.
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np
import gensim

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import get_tokenized_lemmas_without_stopwords

from keras.preprocessing.text import text_to_word_sequence


def create_bow(articles, headlines, vocab, stop_words='english', binary=False):
    hcv = TfidfVectorizer(vocabulary=vocab, norm='l2',
                          tokenizer=get_tokenized_lemmas_without_stopwords, binary=binary)
    bcv = TfidfVectorizer(vocabulary=vocab, norm='l2',
                          tokenizer=get_tokenized_lemmas_without_stopwords, binary=binary)
    X_head = hcv.fit_transform(articles).toarray()
    X_body = bcv.fit_transform(headlines).toarray()

    print(X_head.shape, X_body.shape, np.hstack((X_head, X_body)).shape)

    return np.hstack((X_body, X_head))


def load_word2vec():
    google_vec = gensim.models.KeyedVectors.load_word2vec_format('../data_sets/GoogleNews-vectors-negative300.bin', binary=True)
    return google_vec


def get_vectors(word_to_vec, text):
    tokens = text_to_word_sequence(text)
    return np.asarray([word_to_vec[token] for token in tokens if token in word_to_vec])


# Given a list of articles/headlines etc computes bigrams
# Returns list of dict, with each entry the bigram count for associated text
def compute_bigrams(text):
    bigrams=[]
    count=0
    for j in text:
        if count == 5:
            break
        else:
            count +=1
        words={}
        info=j.split(' ')
        for i in range(len(info)-1):
            if info[i]==' ' or info[i+1]==' ':
                continue
            word_pair=' '.join(info[i:i+2])
            if words.has_key(word_pair):
                words[word_pair] = words[word_pair]+1
            else:
                words[word_pair]=1
        bigrams.append(words)
    return bigrams


def test_word_to_vec_feature():
    gv = load_word2vec()
    print("Successfully loaded word2vec")
    vecs = get_vectors(gv, "This is a sample headline")
    print("Vecs", vecs)
    print("Successfully got vectors")

if __name__ == "__main__":
    test_word_to_vec_feature()
