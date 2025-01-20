# Importing librairies
from read_cifar import *
import numpy as np
import matplotlib.pyplot as plt
import os

def dists(A,B):
    # The distance between every test observation and training observaztion
    norm_A = np.sum(A**2, axis=1, keepdims=True)
    norm_B = np.sum(B**2, axis=1, keepdims=True)
    dot_products = np.dot(A, B.T)
    dists = np.sqrt(norm_A - 2 * dot_products + norm_B.T)
    return dists


def k_smallest_numbers(L,k):
    # Returning the indexes of the smallest k values in the list L
    dict_L = {i : L[i] for i in range(len(L))}
     # Sorting the dictionary based on values
    sorted_items = sorted(dict_L.items(), key=lambda x: x[1])

    # Creating a new dictionary from the sorted items
    sorted_indexes = list(dict(sorted_items).keys())

    return sorted_indexes[:k]


def most_occured_class(L):
    # Returning the most occured class

    # We initiate a dictionnary that takes as keys the counts of elements in the list L, and as values a list of the elements having this count
    dict_counts = dict()

    # Creating the dictionnary
    for el in L : 
        count_el = np.count_nonzero(L==el)
        if count_el not in dict_counts.keys():
            dict_counts[count_el] = [el]
        else : 
            dict_counts[count_el].append(el)
    
    # We sort the keys 
    sorted_keys = sorted(dict_counts.keys(),reverse=True)
    print(sorted_keys)
    # We take the class that has more counts, if more than 1 class has the maximum count, we choose randomly one of the classes
    most_occured_class = dict_counts[sorted_keys[0]][0]

    return most_occured_class


def knn_predict(dists, labels_train, k):
    # Predicting the class of the elements in the data_test based on the KNN method
    m = len(dists[0])

    predicted_classes = []
    for i in range(m):
        # For every element in the test_observation
        distance_i = dists[:,i]

        # We find the indexes of the kNN in the training data
        indexes = k_smallest_numbers(distance_i,k)

        # We find their labels
        labels = labels_train[indexes]

        # We attribute the most occured label to our test_observation
        predicted_class = most_occured_class(labels)
        predicted_classes.append(predicted_class)

    # We return the list of the predicted classes
    return predicted_classes

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):
    # Evaluating the performance of the kNN method by its accuracy

    # We calculate the distances between test observations and training observations
    distances = dists(data_train, data_test)
    # We predict the labels based on the kNN
    labels_predicted = knn_predict(distances, labels_train, k)

    m = len(labels_test)

    # We calculate the number of valid predictions
    nbr_right_predictions = 0
    for i in range(m):
        if labels_predicted[i] == labels_test[i]:
            nbr_right_predictions +=1
    # we calculate the accuracy
    accuracy = nbr_right_predictions/m
    return accuracy

if __name__ == '__main__' :
    # Question 4 : 
    ## Pour cette question, j'ai opté à ne pas utiliser la fonction evaluate_knn
    ## car le calcul de la distance entre les deux matrices prend beaucoup de temps 
    ## et en utilisant cette fonction, il faut que je la recalcule 20 fois, et de trouver
    ## à chaque fois les k plus proches élements alors que je peux trouver directement les 20
    ## proches avec un seul calcul et en déduire les k proches pour k inférieur à 20.
    ## Donc j'ai opté à utiliser l'approche suivante

    # Percentage of the training data
    split_factor = 0.9

    # Reading and splitting the data
    data,labels = read_cifar('data/cifar-10-batches-py/')
    data_train, labels_train, data_test, labels_test = split_dataset(data,labels,split_factor)
    
    # Calculating the distace matrix one time
    distances = dists(data_train,data_test)

    #Initializing my list of k and my accuracy list
    k_list = [k for k in range(1,21)]
    accuracy_list = [0 for k in range(20)]
    m = len(labels_test)
    for i in range(m):
        # For every observation in the test observations
        print(i)
        # We calculate the 20NN 
        closest_20 = k_smallest_numbers(distances[:,i],20)

        # We then conclude the kNN for k smaller than 20
        for k in range(1,len(closest_20)+1):
            # We predict the class of the observation using the kNN method
            predicted_class = most_occured_class(labels_train[closest_20[:k]])
            # We add 1/m to its valid predictions if the prediction is valid
            if predicted_class == labels_test[i]:
                accuracy_list[k-1] = accuracy_list[k-1] + 1/m
            # Finally, we'll get the cardinal of the right predictions divided by m which represents the accuracy

    plt.plot(k_list, accuracy_list, 'o', linestyle = '')
    plt.xlabel('Values of k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy depending on k')
    file_path = os.path.join('results', 'kNN-accuracies.png')
    plt.savefig(file_path)
    plt.show()