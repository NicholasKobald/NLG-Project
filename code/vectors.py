#
#
# WIP
# this might not need it's own file.
#will need to download the google word2vec 

import warnings
#FIXME hopefully something I can do about this, because it's a bit ugly
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim


class Vectors(object):

    def __init__(self):
        model = gensim.models.KeyedVectors.load_word2vec_format('../data_sets/GoogleNews-vectors-negative300.bin', binary=True)


def test_vectors():
    print("Loading google word2vecs")
    v = Vectors()
    print("Successfully loaded word2vec")

if __name__ == "__main__":
    v = test_vectors()
