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
        output_list = []
        with open(file, newline='') as csvfile:
            # On saute la première ligne (Contenant les noms des colonnes)
            next(csvfile)
            spamreader = csv.reader(csvfile, delimiter=';')
            for row in spamreader:
                output_list.append(row)
            # Renvoi la liste contenant le csv
            return output_list

    # Standardise la liste en fonction des indices
    def standardize(self, list, indices):
        # Pour chaque colonne, on calcule la moyenne et l'écart type et on standardise la colonne
        # Boucle de parcours des colonnes
        for i in range(len(list[0])) :
            # Boucle de parcours des indices
            for k in range(len(indices)):
                if i+1==indices[k] :
                    moyenne = 0
                    ecart_type = 0
                    # Calcul de la moyenne pour chaque catégorie
                    for j in range(len(list)):
                        moyenne += list[j][i]
                    moyenne = moyenne/len(list)
                    # Calcul des écarts types
                    for j in range(len(list)):
                        ecart_type += (list[j][i]-moyenne)**2
                    ecart_type = math.sqrt(ecart_type/len(list))
                    # Affectation de la valeur standardisée
                    for j in range(len(list)):
                        list[j][i] = (list[j][i]-moyenne)/ecart_type

    # Mise à l'échelle (Codée pour tests)
    def rescaling(self, list, indices):
        for i in range(14):
            for k in range(len(indices)):
                if i+1==indices[k]:
                    for j in range(len(list)):
                        list[j][i] = ((list[j][i]-1)/99)*9+1

    # Extrait la caractéristique à l'emplacement Indice
    def IndiceExtract(self, list_input, indice):
        list_output = []
        for i in range(len(list_input)):
            # On met l'indice de target dans la nouvelle liste
            list_output.append(list_input[i][indice-1])
            # On supprime l'élément
            del(list_input[i][indice-1])
        return list_output

    def DataToNumber(self, list):
        # Transformation du tableau en int ou float selon besoin
        for i in range(len(list[0])):
            for j in range(len(list)):
                # Si c'est pas un int alors on le change en float
                try:
                    list[j][i] = int(list[j][i])
                except:
                    list[j][i] = float(list[j][i])

    def SplitList(self, list, split_number):
        list_one = []
        list_two = []
        # On sépare la liste de 0 jusqu'au split
        for i in range(split_number):
            list_one.append(list[i])
        # On copie le reste dans la 2e liste
        for i in range(len(list_one),len(list)):
            list_two.append(list[i])
        return list_one, list_two

    # Split la liste en petits batch de taille size au fur et à mesure (4 dans notre cas)
    def batch_split(self, list, size):
        # On parcourt la liste 4 par 4
        for i in range(0, len(list), size):
            # Crée un index de taille size dans la liste
            yield list[i:i+size]

class NeuralNet:

    def __init__(self, batch_size, nb_classes, file):
        self.data = Computing().reader(file) # CSV
        self.batch_size = batch_size # Taille des lots
        self.nb_classes = nb_classes # Nombre de classes
        # On crée une seed random pour randomiser plus les poids
        self.seed = np.random.choice(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        np.random.seed(self.seed)
        # On randomise les poids de départ des matrices de poids entre les couches
        self.layer1_weights = np.random.rand(13, 10) # Matrice de poids
        self.layer2_weights = np.random.rand(10, 1) # Matrice de poids
        self.taux_apprentissage = 0.01 # Taux d'apprentissage
        # data = (303 instances, 14 catégories)
        self.nb_instances = len(self.data)
        self.nb_features = len(self.data[0])
        # Transforme toutes les données en float ou int
        Computing().DataToNumber(self.data)
        # Shuffle des données
        random.shuffle(self.data)

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

    def training(self, list_input, list_output, number_epoch):
        # On crée des batch des données de taille 4 qui changeront au fur et à mesure
        batch_input = list(Computing().batch_split(list_input, self.batch_size))
        batch_output = list(Computing().batch_split(list_output, self.batch_size))

        # On entraîne le réseau un certain nombre de fois
        for i in range(number_epoch):
            # On l'entraîne batch par batch
            for j in range(len(batch_input)):

                # On récupère les matrices qu'on passera dans le réseau
                layer1_weights_output, layer2_weights_output = self.testing(batch_input[j])

                # On calcule l'erreur et on rétropopage en réajustant les poids
                # D'abord l'erreur et le delta de la dernière couche
                layer2_error = batch_output[j]-layer2_weights_output
                layer2_delta = layer2_error*self.__sigmoid_derivative(layer2_weights_output)
                layer2_adjustment = layer1_weights_output.transpose().dot(layer2_delta)
                # Puis la couche précédente
                layer1_error = layer2_delta.dot(self.layer2_weights.transpose())
                layer1_delta = layer1_error*self.__sigmoid_derivative(layer1_weights_output)
                layer1_adjustment = batch_input[j].transpose().dot(layer1_delta)
                # On met à jour les poids d'entrées en fonction de l'ajustement et du taux d'apprentissage
                self.layer1_weights += layer1_adjustment*self.taux_apprentissage
                self.layer2_weights += layer2_adjustment*self.taux_apprentissage

    def testing(self, input):
        # On calcule les poids de la 2e couche en multipliant la matrice d'entrée et la matrice de poids
        # On applique la fonction Sigmoid sur chaque résultat
        layer1_output = self.__sigmoid(dot(input, self.layer1_weights))
        layer2_output = self.__sigmoid(dot(layer1_output, self.layer2_weights))
        return layer1_output, layer2_output

if __name__ == "__main__":

    nn = NeuralNet(4, 1, 'heart_disease_dataset.csv')

    # Séparation des données en 75/25
    split = int(.75*nn.nb_instances)
    training_data_input, testing_data_input = Computing().SplitList(nn.data, split)

    # Standardisation + Extraction de l'attribut target pour chaque jeu de données
    Computing().standardize(training_data_input, [1,4,5,8,10])
    training_data_output = Computing().IndiceExtract(training_data_input, 14)
    Computing().standardize(testing_data_input, [1,4,5,8,10])
    testing_data_output = Computing().IndiceExtract(testing_data_input, 14)

    # On cast les listes en array pour pouvoir utiliser les fonctions sur les matrices de Numpy
    training_data_input = array(training_data_input)
    # On transpose la matrice d'output (Target) afin de faciliter l'utilisation des fonctions de Numpy
    training_data_output = array([training_data_output]).transpose()

    # On entraîne le réseau le nombre de fois qu'on veut (Nombre d'époques)
    nn.training(training_data_input, training_data_output, 500)

    # Initialisation des données nécessaires pour les calculs
    error_sum = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    error_number = 0

    # Lancement du réseau sur les données de test
    for i in range(len(testing_data_input)):
        # On récupère la sortie trouvée dans Output
        initial_weight, output = nn.testing(array(testing_data_input[i]))

        # On calcule les différentes données
        # On arrondi le résultat trouvé par le réseau neuronal
        if (round(output[0],0) != testing_data_output[i]):
            error_number += 1
        if (round(output[0],0) == 0 and testing_data_output[i] == 0):
            true_negative += 1
        if (round(output[0],0) == 1 and testing_data_output[i] == 0):
            false_positive += 1
        if (round(output[0],0) == 1 and testing_data_output[i] == 1):
            true_positive += 1
        if (round(output[0],0) == 0 and testing_data_output[i] == 1):
            false_negative += 1

    error_percentage = (false_positive+false_negative)/len(testing_data_input)*100
    print("")
    print("Vrai positif : ",true_positive)
    print("Vrai négatif : ",true_negative)
    print("Faux positif : ",false_positive)
    print("Faux négatif : ",false_negative)
    print("")
    print("Pourcentage d'erreurs : ", error_percentage)
    print("Sensibilité : ", true_positive/(true_positive+false_negative))
    print("Précision : ", true_positive/(true_positive+false_positive))
    print("Perf globale : ", (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative))
    print("")
    print("Seed actuelle : ",nn.seed)
