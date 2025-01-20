# Importing librairies
import pickle 
import numpy as np
import random as rd

def read_cifar_batch(file):
    # Reading a batch in the cifar_data_batches and returning features and labels in a tuple
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    data = np.array(dict[b'data'],dtype = np.float32)

    labels = np.array(dict[b'labels'],dtype = np.int64)
    return data,labels

def read_cifar(path):
    # Reading and joining all the cifar_batch data

    # We initiate our joining with the test_data
    data, labels = read_cifar_batch(path +'test_batch')

    # This list contains all the directories of the training files
    training_files = [ path + 'data_batch_' + str(i) for i in range(1,6) ]

    # We append then every training data into our data list
    for file in training_files:
       data_training_file, labels_training_file = read_cifar_batch(file)
       data = np.append(data, data_training_file, axis = 0)
       labels = np.append(labels, labels_training_file, axis = 0)
    return data,labels

def split_dataset(data,labels,split):
    # Splitting data into training and testing set
    all_lines = [i for i in range(len(data))]
    m = int(split*(len(data)))

    # This line takes m random integer numbers between 0 and len(data)-1
    training_lines = rd.sample(all_lines, m)

    # Here we take into account numbers that weren't taken
    test_lines = []
    for el in all_lines:
        if el not in training_lines:
            test_lines.append(el)
    
    # Our training data is thus the data which the integers were taken randomly
    training_data = data[training_lines]

    training_labels = labels[training_lines]

    #Our test data are the observations that weren't taken in the training data
    test_data = data[test_lines]
    test_labels = labels[test_lines]

    return training_data, training_labels, test_data, test_labels
    
