# 
# Lewis Hunton | 11135261 | 07/10/25 | Electronics Engineering 3rd Year Project
# Neural Network for the identification of hand written 8x8 resolution numbers for results verification of VLSI hardware 
#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import math
import random

torch.manual_seed(6)

#Constant definitions
inputNodes = 64
hiddenNodes = 64
outputNodes = 10

learningRate = 0.1
alpha = 1

hiddenNodesWeights = [[random.random() for _ in range(inputNodes)] for _ in range(hiddenNodes)]
outputNodesWeights = [[random.random() for _ in range(hiddenNodes)] for _ in range(outputNodes)]

outputNodesBias = [0]*outputNodes

digits = load_digits()

#Function definitions
def ELU(x):
    return x if x > 0 else alpha*math.exp(x) - 1

def normalise(x):
    return x/16.0

def startTraining():
    hiddenNodesWeights = torch.rand(inputNodes,hiddenNodes)
    outputNodesWeights = torch.rand(hiddenNodes,outputNodes)

#Data processing definitions
def dataPreprocessing(x,y):
    normalisedX = normalise(x)

    tempTensorx = torch.FloatTensor(normalisedX)
    tempTensory = torch.LongTensor(y)

    return tempTensorx, tempTensory

#Main
def main():
    startTraining()

    print("Available keys:", digits.keys())
    print("Data shape:", digits.data.shape)
    print("Target shape:", digits.target.shape)
    print("Feature names:", digits.feature_names)
    print("Target names:", digits.target_names)

    # Access the data and labels
    x = digits.data
    y = digits.target

    #Function for splitting data set up and holding some data points in reserve for data validation  and credibility during report writing
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
    )









    # Display the first image
    plt.figure(figsize=(8, 6))
    plt.imshow(digits.images[0], cmap='gray')
    plt.title(f"Digit: {digits.target[0]}")
    plt.colorbar()
    print(digits.images[0])
    plt.show()

if __name__ == "__main__":
    main()