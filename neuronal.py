from numpy import exp, array, random, dot
import numpy as np
import csv
import math
import random


# Print une liste
def printList(list):
    for row in list:
        print(*row, sep=", ")

# Standardize

class Computing:

    # Renvoi une liste avec les données du CSV
    def reader(self, file):
        elements = []
        with open(file, newline='') as csvfile:
            next(csvfile)
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                elements.append(row)
            return elements

    # Standardise la liste en fonction des indices
    def standardize(self, list, indices):
        for i in range(14) :
            for k in range(len(indices)):
                if i+1==indices[k] :
                    moyenne = 0
                    ecart_type = 0
                    for j in range(len(list)):
                        moyenne += list[j][i]
                    moyenne = moyenne/len(list)
                    for j in range(len(list)):
                        ecart_type += (list[j][i]-moyenne)**2
                    ecart_type = math.sqrt(ecart_type/len(list))
                    for j in range(len(list)):
                        list[j][i] = (list[j][i]-moyenne)/ecart_type

    def rescaling(self, list, indices):
        for i in range(14):
            for k in range(len(indices)):
                if i+1==indices[k]:
                    for j in range(len(list)):
                        list[j][i] = ((list[j][i]-1)/99)*9+1

    def TargetExtract(self, list_input):
        list_output = []
        for i in range(len(list_input)):
            list_output.append(list_input[i][len(list_input[i])-1])
            list_input[i].pop()
        return list_output

    def DataToNumber(self, list):
        # Transformation du tableau en int ou float selon besoin
        for i in range(len(list[0])):
            for j in range(len(list)):
                try:
                    list[j][i] = int(list[j][i])
                except:
                    list[j][i] = float(list[j][i])

    def SplitList(self, list, split_number):
        list_one = []
        list_two = []
        for i in range(split_number):
            list_one.append(list[i])
        for i in range(len(list_one),len(list)):
            list_two.append(list[i])
        return list_one, list_two

class NeuralNet:

    def __init__(self, batch_size, nb_classes, file):
        self.data = Computing().reader(file) #CSV
        self.batch_size = batch_size # Taille des lots
        self.nb_classes = nb_classes # Nombre de classes
        self.layer1_weights = np.random.rand(13, 5) # Matrice de poids
        self.layer2_weights = np.random.rand(5, 1) # Matrice de poids
        self.taux_apprentissage = 0.01

        # data = (303 instances, 14 catégories)
        self.nb_instances = len(self.data)
        self.nb_features = len(self.data[0])

        # Transforme toutes les données en float ou int
        Computing().DataToNumber(self.data)
        # Shuffle des données
        random.shuffle(self.data)
        # NB: Pas de transposition donc x_train[0,0] = 1e neurone,
        #                               x_train[0,1] = 2e neurone

        """
        # Crée 2 sets de liste: 1 avec les catégories (13), l'autre avec les classes (dim=1)
        self.x_train = [[0 for i in range(self.nb_features)] for j in range(self.batch_size)]
        self.y_train = [[0 for i in range(self.nb_classes)] for j in range(self.batch_size)]

        # De même pour les tests
        self.x_test = [[0 for i in range(self.nb_features)] for j in range(self.batch_size)]
        self.y_test = [[0 for i in range(self.nb_classes)] for j in range(self.batch_size)]
        """

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

    # Split la liste en petits batch de taille size au fur et à mesure (4 dans notre cas)
    def batch_split(self, list, size):
        for i in range(0, len(list), size):
            # Crée un index de taille n dans la liste
            yield list[i:i+size]

    def train(self, list_input, list_output, number_epoch):
        # On crée des batch de 4 qui changeront au fur et à mesure
        batch_input = list(self.batch_split(list_input, self.batch_size))
        batch_output = list(self.batch_split(list_output, self.batch_size))
        for i in range(number_epoch):
            for j in range(len(batch_input)):
                layer_1_output, layer_2_output = self.think(batch_input[j])

                layer2_error = batch_output[j] - layer_2_output
                layer2_delta = layer2_error* self.__sigmoid_derivative(layer_2_output)

                layer1_error = layer2_delta.dot(self.layer2_weights.T)
                layer1_delta = layer1_error * self.__sigmoid_derivative(layer_1_output)

                layer1_adjustment = batch_input[j].T.dot(layer1_delta)
                layer2_adjustment = layer_1_output.T.dot(layer2_delta)

                self.layer1_weights += layer1_adjustment*self.taux_apprentissage
                self.layer2_weights += layer2_adjustment*self.taux_apprentissage

    def think(self, input):
        layer1_output = self.__sigmoid(dot(input, self.layer1_weights))
        layer2_output = self.__sigmoid(dot(layer1_output, self.layer2_weights))
        return layer1_output, layer2_output

'''
        self.shuffle_training_data
        # self.load_attrs_labels()

    def create_one_hot(self, k, ):
        pass

    def test_prediction(self):
        pass

    def training_epoch(self):
        pass

    def train(self, nb_epochs):
        pass
'''

if __name__ == "__main__":
    nn= NeuralNet(4, 1, 'heart_disease_dataset.csv')

    # Séparation des données en 75/25
    split = int(.75*nn.nb_instances)
    training_data_input, testing_data_input = Computing().SplitList(nn.data, split)

    # Standardisation + Extraction de l'attribut target pour chaque jeu de données
    Computing().standardize(training_data_input, [1,4,5,8,10])
    training_data_output = Computing().TargetExtract(training_data_input)
    Computing().standardize(testing_data_input, [1,4,5,8,10])
    testing_data_output = Computing().TargetExtract(testing_data_input)

    training_data_input = array(training_data_input)
    training_data_output = array([training_data_output]).T

    nn.train(training_data_input, training_data_output, 100)

    error_sum = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    error_number = 0
    training_data_output = training_data_output.T
    for i in range(len(testing_data_input)):
        hidden_state, output = nn.think(array(testing_data_input[i]))
        error_sum = abs(testing_data_output[i] - output[0])

        if (round(output[0]) != testing_data_output[i]):
            error_number += 1
        if (round(output[0]) == 0 and testing_data_output[i] == 0):
            true_negative += 1
        if (round(output[0]) == 1 and testing_data_output[i] == 0):
            false_positive += 1
        if (round(output[0]) == 1 and testing_data_output[i] == 1):
            true_positive += 1
        if (round(output[0]) == 0 and testing_data_output[i] == 1):
            false_negative += 1


    error_mean = error_sum / len(testing_data_input)
    error_percentage = error_number / len(testing_data_input)

    print(true_positive,
    true_negative,
    false_positive,
    false_negative)
    print("Pourcentage d'erreurs : ", error_percentage*100)
    print("Sensibilité : ", true_positive/(true_positive+false_negative))
    print("Précision : ", true_positive/(true_positive+false_positive))
    print("Perf globale : ", (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative))
