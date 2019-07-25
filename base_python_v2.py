#!/usr/bin/env python3

# DISCLAIMER: code écrit rapidement, incomplet et potentiellement faux, à ne pas utiliser si vous n'êtes pas soit :
# 1. en groupe CMI
# 2. très à l'aise en python3
# 3. prêt à travailler chez vous en posant des questions à clement.schreiner@unistra.fr
# -- Clément Schreiner

import csv
import numpy as np

def import_data(filename):
    with open(filename, 'r') as f:
        return list(csv.reader(f))


class NNLib:
    """
    Activation function, etc.
    Most methods found in the java code are found in the numpy library and will be easier to use.
    """

    def __init__(self):
        pass


class NeuralNet:
    def __init__(self, filename, batch_size, nb_classes):
        self.data = np.array(import_data(filename))
        self.batch_size = batch_size
        self.nb_classes = nb_classes

        # data.shape = (150, 5)
        self.nb_instances, self.nb_features = self.data.shape


        self.training_size = int(.75*self.nb_instances)
        self.testing_size = self.nb_instances - self.training_size

        # split the dataset into two arrays
        self.training_data = self.data[:self.training_size, ...]
        self.testing_data = self.data[self.training_size:, ...]


        # weight matrices

        self.w_1 = np.random.rand(3,4)
        self.b_1 = np.zeros((3,1))

        self.w_2 = np.random.rand(3,3)
        self.b_2 = np.zeros((3,1))



        # create two sets of array: one with features (dim=4), the other with classes (dim=1)
        self.x_train, self.y_train = np.hsplit(self.training_data, [4])

        # tranpose so that columns are instances
        self.x_train.transpose((1,0))
        self.y_train.transpose((1,0))

        self.x_test, self.y_test = np.hsplit(self.testing_data, [4])
        self.x_test.transpose((1,0))
        self.y_test.transpose((1,0))

        self.shuffle_training_data
        # self.load_attrs_labels()



    def shuffle_training_data(self):
        """ Shuffle the data in-place """
        np.random.shuffle(self.training_data)

    def load_attrs_labels(self, batch):
        pass


    def create_one_hot(self, k, ):
        pass

    def test_prediction(self):
        pass

    def training_epoch(self):
        pass

    def train(self, nb_epochs):
        pass

if __name__ == "__main__":
    nn = NeuralNet('iris_num.data', 4, 3)
    nn.train(800)

    # print(nn.w_1)
    # print(nn.b_1)

    print(nn.x_train[:4]) # print the first four training vectors

    # print(nn.testing_data.shape)
    # print(nn.training_data.shape)


