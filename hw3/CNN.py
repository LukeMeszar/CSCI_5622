
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.utils import to_categorical

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epochs = 12, batch_size=32):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epochs = epochs

        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        self.train_x_shape = train_x.shape
        self.test_x_shape = test_x.shape
        self.train_y = to_categorical(train_y,10)
        self.train_x = train_x
        self.train_x.shape = (self.train_x_shape[0],int(np.sqrt(self.train_x_shape[1])),int(np.sqrt(self.train_x_shape[1])),1)
        self.test_y = to_categorical(test_y,10)
        self.test_x = test_x
        self.test_x.shape = (self.test_x_shape[0],int(np.sqrt(self.test_x_shape[1])),int(np.sqrt(self.test_x_shape[1])),1)


        # normalize data to range [0, 1]
        self.train_x = train_x.astype('float32')
        self.test_x = test_x.astype('float32')

        # TODO: one hot encoding for train_y and test_y

        # TODO: build you CNN model

        self.model = Sequential()
        self.model.add(Conv2D(32,6,6,activation='relu',input_shape=(int(np.sqrt(self.train_x_shape[1])),int(np.sqrt(self.train_x_shape[1])),1)))
        #self.model.add(Conv2D(64, 3, 3,activation='relu'))
        self.model.add(MaxPool2D(pool_size=(4,4)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        #self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))


        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y,
          batch_size=self.batch_size, nb_epoch=self.epochs, verbose=1,shuffle=True)

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''

        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    print("after data")

    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print("acc!!!!!", acc)
