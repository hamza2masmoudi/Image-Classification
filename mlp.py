# Importing librairies
import numpy as np
from read_cifar import *
import matplotlib.pyplot as plt
import os

N = 30  # number of input data
d_in = 3  # input dimension
d_h = 3  # number of neurons in the hidden layer
d_out = 2  # output dimension (number of neurons of the output layer)

def init(d_in, d_h, d_out):
    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises
    
    return w1, b1, w2, b2

data = np.random.rand(N, d_in)  # create a random data
targets = np.random.rand(N, d_out)  # create a random targets



def softmax(x, derivate = False):

    if derivate==False:
        return np.exp(x)/np.exp(np.array(x)).sum(keepdims=True, axis=1) # softmax activation function
    else:
        return x*(1-x) # softmax derivative function

def sigmoid(x):
    # Returning sigmoid
    return 1 / (1 + np.exp(-x))

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    # Upgrading the weights and biases

    z1 = np.dot(data, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    y = sigmoid(z2)

    # Compute loss
    loss = np.mean((y - targets)**2)

    # Backward pass
    delta_z2 = 2 * (y - targets) * y * (1 - y)
    delta_w2 = np.dot(a1.T, delta_z2)
    delta_b2 = np.sum(delta_z2, axis=0, keepdims=True)

    delta_a1 = np.dot(delta_z2, w2.T)
    delta_z1 = delta_a1 * a1 * (1 - a1)
    delta_w1 = np.dot(data.T, delta_z1)
    delta_b1 = np.sum(delta_z1, axis=0, keepdims=True)

    # Update weights and biases
    w1 -= learning_rate * delta_w1
    b1 -= learning_rate * delta_b1
    w2 -= learning_rate * delta_w2
    b2 -= learning_rate * delta_b2

    return w1, b1, w2, b2, loss

def one_hot(labels):
    num_classes = len(np.unique(labels))
    return np.eye(num_classes)[labels]

def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-15  # Small constant to avoid log(0)
    loss = -np.mean(targets * np.log(predictions + epsilon) + (1 - targets) * np.log(1 - predictions + epsilon))
    return loss
def accuracy(A,B):
    c=0
    for i in range(len(A)):
        if A[i] == B[i]: 
            c+=1
    return c/len(A)
def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    z1 = np.matmul(data, w1) +b1
    a1 = sigmoid(z1,derivate=False)    # Sigmoid activation

    # Implementing feed forward propagation on output layer
    z2 = np.matmul(a1, w2) +b2
    a2 = softmax(z2,derivate=False)    # Softmax activation

    # Backpropagation phase
    # Updating the W2 and b2
    e2 = a2-targets
    dw2 = e2 *softmax(a2,derivate=True)
    new_w2 = np.dot(a1.T, dw2) / N
    new_b2 = (1/a1.shape[1])*dw2.sum(axis=0, keepdims=True)

    # Updating the W1 and b1
    e1 = np.dot(dw2, w2.T)
    dw1 = e1 * sigmoid(a1,derivate=True)
    new_w1 = np.dot(data.T, dw1) / N
    new_b1 = (1/data.shape[1])*dw1.sum(axis=0, keepdims=True)

    # Gradient descent
    w2 = w2 - learning_rate * new_w2
    w1 = w1 - learning_rate * new_w1
    b2 = b2 - learning_rate * new_b2
    b1 = b1 - learning_rate * new_b1

    # Compute loss (Binary Cross Entropy)
    loss = binary_cross_entropy_loss(a2,targets)

    return w1,b1,w2,b2,loss

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
     # Forward pass
    # Implementing feedforward propagation on hidden layer
    z1 = np.matmul(data_train, w1) +b1
    a1 = sigmoid(z1,derivate=False)

    # Implementing feed forward propagation on output layer
    z2 = np.matmul(a1, w2) +b2
    a2 = softmax(z2,derivate=False)
    N = data_train.shape[0]

    # Backpropagation phase
    # Updating the W2 and b2
    e2 = a2-targets
    dw2 = e2 *softmax(a2,derivate=True)
    new_w2 = np.dot(a1.T, dw2) / N
    new_b2 = (1/a1.shape[1])*dw2.sum(axis=0, keepdims=True)

    # Updating the W1 and b1
    e1 = np.dot(dw2, w2.T)
    dw1 = e1 * sigmoid(a1,derivate=True)
    new_w1 = np.dot(data.T, dw1) / N
    new_w2 = (1/data.shape[1])*dw1.sum(axis=0, keepdims=True)

    # Gradient descent
    w2 = w2 - learning_rate * new_w2
    w1 = w1 - learning_rate * new_w1
    b2 = b2 - learning_rate * new_b2
    b1 = b1 - learning_rate * new_w2

    # Clculating the error
    loss= binary_cross_entropy_loss(a2,targets)

    # Calculating the accuracy
    train_accuracies = accuracy(a2, labels_train)

    return w1,b1,w2,b2,loss,train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    # Forward pass
    # Implementing feedforward propagation on hidden layer
    z1 = np.matmul(data_test, w1) +b1
    a1 = sigmoid(z1,derivate=False)

    # Implementing feed forward propagation on output layer
    z2 = np.matmul(a1, w2) +b2
    a2 = softmax(z2,derivate=False)

    # Compute the testing accuracy
    test_accuracy = accuracy(a2, labels_test)

    return test_accuracy 


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    d_in = data_train.shape[1]   # input dimension
    d_out = np.unique(labels_train).shape[0]  # output dimension : 10 classes
    train_accuracy=[]
    w1, b1, w2, b2=init(d_in,d_h,d_out)
    for i in range(num_epoch):
        w1, b1, w2, b2, loss, train_accuracies=train_mlp(w1, b1, w2, b2, data_train, one_hot(labels_train), learning_rate, i)
        test_accuracy=test_mlp(w1,b1,w2,b2,data_test,one_hot(labels_test))
        train_accuracy.append(train_accuracies)
        print("Epoch {}/{}".format(i+1,num_epoch))
        print("Train_Accuracy : {}         Test_Accuracy : {}".format(round(train_accuracies,6),round(test_accuracy,6)))
    return train_accuracy, test_accuracy

if __name__ == '__main__':
    split_factor = 0.9
    d_h =64
    learning_rate = 0.1
    num_epoch = 100
    file_path = os.path.join('results', 'kNN-accuracies.png')
    plt.savefig(file_path)
    plt.show()