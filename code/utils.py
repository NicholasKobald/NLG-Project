#The code based on baseline provided by the FNC organization,
#under the the Apache License
#https://github.com/FakeNewsChallenge/fnc-1-baseline
import string
from csv import DictReader

import nltk
from nltk.corpus import stopwords

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import os

STOP_WORDS = set(stopwords.words('english'))


class DataSet():

    def __init__(self,
                 bodies_fname="train_bodies.csv",
                 stance_fname="train_stances.csv",
                 path="../data_sets"
                ):

        self.path = path

        bodies = "train_bodies.csv"
        stances = "train_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)

        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        self.articles = dict()
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        self.create_article_headline_stance_triples()

    def read(self, filename):
        rows = []
        with open(self.path + "/" + filename, "r",  encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows

    #@return string
    def parse_article(self, article):
        return ' '.join([x.lower() for x in article.split() if x not in STOP_WORDS])

    def print_stances(self, print_limit=10):
        print("First", print_limit, "stances")
        for i in range(print_limit):
            print(self.stances[i])

    def print_articles(self, print_limit=10):
        print("First", print_limit, "articles")
        for i in range(print_limit):
            print(self.articles[self.stances[i]['Body ID']])

    def get_stance_counts(self):
        counts = dict(unrelated=0, discuss=0,
            agree=0, disagree=0
        )
        for s in self.stances:
            counts[s['Stance']] += 1

        return counts

    def create_article_headline_stance_triples(self):
        self.triples = dict(
            stances=[],
            articles=[],
            headlines=[]
        )

        for s in self.stances:
            self.triples['stances'].append(s['Stance'])
            self.triples['articles'].append(self.articles[s['Body ID']])
            self.triples['headlines'].append(s['Headline'])


_wnl = nltk.WordNetLemmatizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_all_stopwords():
    stop_words_nltk = set(stopwords.words('english'))  # use set for faster "not in" check
    stop_words_sklearn = feature_extraction.text.ENGLISH_STOP_WORDS
    all_stop_words = stop_words_sklearn.union(stop_words_nltk)
    return all_stop_words


def get_tokenized_lemmas_without_stopwords(s, stop_words=get_all_stopwords()):
    return [normalize_word(t) for t in nltk.word_tokenize(s)
            if t not in string.punctuation
            and t.lower() not in stop_words]


def generate_vocab(dataset, size=5000, stop_words=None):
    cv = CountVectorizer(max_features=size, tokenizer=get_tokenized_lemmas_without_stopwords)
    cv.fit(dataset['Headline'] + dataset['articleBody'])
    return cv.vocabulary_


def transform_text(t, vocab, max_len):
    tokens = get_tokenized_lemmas_without_stopwords(t)
    unk_token = vocab['unk']
    res = np.full(max_len, unk_token.index)

    for i in range(min(max_len, len(tokens))):
        res[i] = vocab.get(tokens[i], vocab['unk']).index

    return res


def gen_or_load_feats(generator, feature_file):
    if not os.path.isfile(feature_file):
        feats = generator()
        np.save(feature_file, feats)

    return np.load(feature_file)

