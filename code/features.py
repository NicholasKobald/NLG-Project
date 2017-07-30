import numpy as np


def create_bow(articles, headlines, tokenizer, mode='binary'):
    bodies_bow = tokenizer.texts_to_matrix(articles, mode=mode)
    heads_bow = tokenizer.texts_to_matrix(headlines, mode=mode)
    return np.hstack((bodies_bow, heads_bow))
