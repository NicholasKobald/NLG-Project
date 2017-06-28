import numpy

from keras.layers import Input, LSTM, Dense
from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding

from utils import DataSet

numpy.random.seed(7)

def main():
    train_set = DataSet()
    test_set = DataSet('competition_test_bodies.csv',
                       'competition_test_stances.csv')

    embedding_vector_length = 32
    model = Sequential()
    #model.add()


if __name__ == "__main__":
    main()
