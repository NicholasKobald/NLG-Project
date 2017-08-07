import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils import get_tokenized_lemmas_without_stopwords


def create_bow(articles, headlines, vocab, stop_words='english', binary=False):
    hcv = TfidfVectorizer(vocabulary=vocab, norm='l2',
                          tokenizer=get_tokenized_lemmas_without_stopwords, binary=binary)
    bcv = TfidfVectorizer(vocabulary=vocab, norm='l2',
                          tokenizer=get_tokenized_lemmas_without_stopwords, binary=binary)
    X_head = hcv.fit_transform(articles).toarray()
    X_body = bcv.fit_transform(headlines).toarray()

    print(X_head.shape, X_body.shape, np.hstack((X_head, X_body)).shape)

    return np.hstack((X_body, X_head))
