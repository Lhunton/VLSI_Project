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
import math
import random

torch.manual_seed(6)

#Constant definitions
inputNodes = 64
hiddenNodes = 64
outputNodes = 16

learningRate = 0.1
alpha = 1

hiddenNodesWeights = [[0]*inputNodes]*hiddenNodes
outputNodesWeights = [[0]*hiddenNodes]*outputNodes

outputNodesBias = [0]*outputNodes

digits = load_digits()

#Function definitions
def ELU(x):
    return x if x > 0 else alpha*math.exp(x) - 1

def startTraining():
    for x in range (0,inputNodes):
        for y in range (0,hiddenNodes):
            hiddenNodesWeights[x][y] = random.random()

    for n in range (0,hiddenNodes):
        for o in range (0,outputNodes):
            outputNodesWeights[o][n] = random.random()

def main():
    """Main function"""

    startTraining()

    print("Available keys:", digits.keys())
    print("Data shape:", digits.data.shape)
    print("Target shape:", digits.target.shape)
    print("Feature names:", digits.feature_names)
    print("Target names:", digits.target_names)

    # Access the data and labels
    X = digits.data
    y = digits.target

    # Display the first image
    plt.figure(figsize=(8, 6))
    plt.imshow(digits.images[0], cmap='gray')
    plt.title(f"Digit: {digits.target[0]}")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()