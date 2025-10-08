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

learningRate = 0.4
alpha = 1
batchSize = 32

digits = load_digits()

#Function definitions
def ELU(x):
    return torch.where(x > 0, x, alpha * torch.exp(x) - 1)

def ELUDerivative(x):
    return torch.where(x > 0, torch.ones_like(x), alpha * torch.exp(x))

def normalise(x):
    return x/16.0

def startTraining():
    hiddenNodesWeights = torch.rand(inputNodes,hiddenNodes)
    outputNodesWeights = torch.rand(hiddenNodes,outputNodes)
    hiddenNodesBias = torch.zeros(hiddenNodes)
    outputNodesBias = torch.zeros(outputNodes)
    return hiddenNodesWeights,outputNodesWeights,hiddenNodesBias,outputNodesBias

#Data processing definitions
def dataPreprocessing(x,y):
    normalisedX = normalise(x)

    tempTensorx = torch.FloatTensor(normalisedX)
    tempTensory = torch.LongTensor(y)

    return tempTensorx, tempTensory

def dataloading(trainX, trainY, testX, testY):
    #convert tensors into pairs of input + output tensors
    trainingDataset = TensorDataset(trainX, trainY)
    testingDataset = TensorDataset(testX,testY)

    #convert individual data tensors into a larger dataset (loaders)
    trainingLoader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
    testingLoader = DataLoader(testingDataset, batch_size=batchSize, shuffle=True)

    return trainingLoader, testingLoader

#Forward propagation definitions
def forwardPropagation(inputs, hiddenWeights, outputWeights, hiddenBias, outputBias):
    #matrix multiplication of hidden layer
    hiddenInputs = torch.matmul(inputs, hiddenWeights) + hiddenBias
    hiddenOutputs = torch.zeros_like(hiddenInputs)

    #Activation function applications
    for i in range(hiddenInputs.shape[0]):
        for j in range(hiddenInputs.shape[1]):
            hiddenOutputs[i,j] = ELU(hiddenInputs[i,j])

    #matrix multiplication of output layer
    outputInputs = torch.matmul(hiddenOutputs, outputWeights) + outputBias

    return outputInputs, hiddenOutputs

#Training loop definitions
def trainingLoop(loader, hiddenWeights, outputWeights, hiddenBias, outputBias):
    #Loop through batches with forward pass
    for batch_idx, (data, targets) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"Input data shape: {data.shape}")  # [32, 64]
        print(f"Targets shape: {targets.shape}")  # [32]

        outputs, hiddenActivations = forwardPropagation(data, hiddenWeights, outputWeights, hiddenBias, outputBias)

        print(f"Hidden activations shape: {hiddenActivations.shape}")  # [32, 64]
        print(f"Output logits shape: {outputs.shape}")  # [32, 10]
        print(f"Sample output: {outputs[0]}")

#Loss calculations
def loss(outputs, targets):
        return F.cross_entropy(outputs, targets)

#Backward propagation definitions
def backwardsPropagation(inputs, hiddenOutputs, outputs, targets, hiddenWeights, outputWeights, hiddenBias, outputBias):


#Main
def main():
    # Access the data and labels
    x = digits.data
    y = digits.target

    #Function for splitting data set up and holding some data points in reserve for data validation  and credibility during report writing
    trainX, testX, trainY, testY = train_test_split(
    x, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    trainingTensorsx, trainingTensorsy = dataPreprocessing(trainX, trainY)
    testingTensorsx, testingTensorsy = dataPreprocessing(testX, testY)

    trainingLoader, testLoader = dataloading(trainingTensorsx, trainingTensorsy, testingTensorsx, testingTensorsy)

    hiddenWeights, outputWeights, hiddenBias, outputBias = startTraining()

    trainingLoop(trainingLoader, hiddenWeights, outputWeights, hiddenBias, outputBias)


    # Display the first image
    plt.figure(figsize=(8, 6))
    plt.imshow(digits.images[0], cmap='gray')
    plt.title(f"Digit: {digits.target[0]}")
    plt.colorbar()
    print(digits.images[0])
    plt.show()

if __name__ == "__main__":
    main()