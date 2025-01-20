# Image Classification



# Image Classification: CIFAR-10 Dataset

## Description

The CIFAR-10 dataset comprises 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The dataset is divided into five training batches and one test batch, each containing 10,000 images. The test batch contains exactly 1,000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some batches may have more images from one class than another. Together, the training batches have exactly 5,000 images from each class.

The CIFAR-10 dataset includes the following 10 classes:
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

![CIFAR-10 Classes](cifar.JPG)

## Dataset Download

[Data CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## Project Content

### Prepare the Dataset

Create a Python file named _read_cifar.py_ that reads data in all batches, including the test batch, and splits it into a training set and a test set. This file contains the following functions:
- read_cifar_batch
- read_cifar
- split_dataset

### k-nearest neighbors

Create a Python file named _knn.py_ to train a KNN model, predict the test dataset, and evaluate the performance. This file contains the following functions:
- distance_matrix
- knn_predict
- evaluate_knn
- plot_KNN

### Artificial Neural Network

Create a Python file named _mlp.py_ to develop a classifier based on a multilayer perceptron (MLP) neural network. This file contains the following functions:
- learn_once_mse
- learn_once_cross_entropy
- train_mlp
- test_mlp
- run_mlp_training
- plot_ANN
# Visuals
After applying  different algorithms to classify our Dataset Cidar-10, we got the following results that represent the accuracy of each model:
- k-nearest neighbors

![Semantic description of image](Results/knn.png)

- Artificial Neural Network

![Semantic description of image](Results/mlp.png)

