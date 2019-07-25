# -*- coding: utf-8 -*-

from numpy import exp, array, random, dot
import csv
import math
from random import shuffle

class NNLib():
    def LoadData(self, filename):
        elements = []
        with open(filename) as csvfile:
            next(csvfile)
            spamreader = csv.reader(csvfile, delimiter=';')

            for row in spamreader:
                elements.append(row)

            for i in range(len(elements)):
                for j in range(len(elements[i])):
                    try:
                        elements[i][j] = int(elements[i][j])
                    except:
                        elements[i][j] = float(elements[i][j])
            shuffle(elements)
            return elements

    def ComputeNormalize(self, data, indices):
        for indice in range(len(indices)):

            indice_to_test = indices[indice] - 1

            moyenne = 0
            ecart_type = 0

            # calcul de la moyenne
            somme = 0
            for i in range(len(data)):
                somme += data[i][indice_to_test]
            moyenne = somme/len(data)

            # calcul ecart type
            total_ecart = 0
            for i in range(len(data)):
                total_ecart += pow(data[i][indice_to_test] - moyenne, 2)
            ecart_type = math.sqrt(total_ecart/len(data))

            for i in range(len(data)):
                data[i][indice_to_test] = (data[i][indice_to_test] - moyenne)/ecart_type

        return data

    def ExtractTargets(self, data):
        extracted = []
        for i in range(len(data)):
            size = len(data[i]) -1
            extracted.append(data[i][size])
        return extracted

    def RemoveLastColumn(self, data):
        final = []
        for i in range(len(data)):
            data[i].pop()
            final.append(data[i])
        return final



class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, batch_size, nb_classes, hidden_neurons):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.layer1 = NeuronLayer(hidden_neurons, 13)
        self.layer2 = NeuronLayer(1, hidden_neurons)

        self.taux_apprentissage = 0.01
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def chunks(self, l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i+n]

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        batches_t = list(self.chunks(training_set_inputs, self.batch_size))
        batches_o = list(self.chunks(training_set_outputs, self.batch_size))
    
        for iteration in range(number_of_training_iterations):
            for b in range(len(batches_t)):
                # Pass the training set through our neural network
                output_from_layer_1, output_from_layer_2 = self.think(batches_t[b])

                # Calculate the error for layer 2 (The difference between the desired output
                # and the predicted output).

                layer2_error = batches_o[b] - output_from_layer_2
                layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)


                # Calculate the error for layer 1 (By looking at the weights in layer 1,
                # we can determine by how much layer 1 contributed to the error in layer 2).
                layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
                layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

                # Calculate how much to adjust the weights by
                layer1_adjustment = batches_t[b].T.dot(layer1_delta)
                layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

                # Adjust the weights.
                self.layer1.synaptic_weights += layer1_adjustment*self.taux_apprentissage
                self.layer2.synaptic_weights += layer2_adjustment*self.taux_apprentissage

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    '''
    def print_weights(self):
        print "    Layer 1 (5 neuron, with 13 inputs):"
        print self.layer1.synaptic_weights
        print "    Layer 2 (1 neuron, with 5 inputs):"
        print self.layer2.synaptic_weights
    '''

if __name__ == "__main__":

    # Init things
    random.seed(1)
    Lib = NNLib()

    # 1. Load data from dataset
    dataset = Lib.LoadData('heart_disease_dataset.csv')

    # 2. Divize data into two sets, one for training and one for testing
    testing_set = dataset[:len(dataset)//4]
    training_set = dataset[len(dataset)//4:]

    print("total data: ", len(dataset))
    print("training: ", len(training_set))
    print("testing: ", len(testing_set))

    # 3. Normalize data
    indices_to_normalize = [1, 4, 5, 8, 10]
    training_set = Lib.ComputeNormalize(training_set, indices_to_normalize)
    testing_set = Lib.ComputeNormalize(testing_set, indices_to_normalize)

    # Extract targets
    training_outputs = Lib.ExtractTargets(training_set)
    testing_outputs = Lib.ExtractTargets(testing_set)

    # Remove last column
    training_set = Lib.RemoveLastColumn(training_set)
    testing_set = Lib.RemoveLastColumn(testing_set)

    # 4. Create neural network
    neural_network = NeuralNetwork(4, 1, 5)

    # 5. Train neural network
    training_set_inputs = array(training_set)
    training_set_outputs = array([training_outputs]).T

    # Train the neural network using the training set.
    neural_network.train(training_set_inputs, training_set_outputs, 1)

    # 6. Test neural network
    somme_erreurs = 0
    nb_erreurs = 0
    for t in range(len(testing_set)):
        hidden_state, output = neural_network.think(array(testing_set[t]))
        #print("OUTPUT", output[0], testing_outputs[t])
        somme_erreurs += abs(testing_outputs[t] - output[0])
        if(round(output[0]) != testing_outputs[t]):
            nb_erreurs += 1

    moyenne_erreurs = somme_erreurs / len(testing_set)
    pourcentage_erreurs = float(nb_erreurs) / float(len(testing_set))
    #print("MOYENNE ERREUR: ", moyenne_erreurs)
    print("POURCENTAGE ERREURS: ", pourcentage_erreurs*100)
