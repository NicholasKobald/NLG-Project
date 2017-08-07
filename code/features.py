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

#Given a list of articles/headlines etc computes bigrams
#Returns list of dict, with each entry the bigram count for associated text
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