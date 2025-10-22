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
import json

torch.manual_seed(6)

#Constant definitions
inputNodes = 64
hiddenNodes = 64 #101 and 17 are weirdly optimal
outputNodes = 10

learningRate = 0.14
alpha = 1
batchSize = 32
epochs = 50

digits = load_digits()

#Function definitions
#activation functions
activationFunctions = {
    'ELU': (lambda x: torch.where(x > 0, x, alpha * torch.exp(x) - 1),
            lambda x: torch.where(x > 0, torch.ones_like(x), alpha * torch.exp(x))),

    'ReLu': (lambda x: torch.maximum(torch.zeros_like(x), x),
             lambda x: torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))),

    'Sigmoid': (lambda x: (1/(1+torch.exp(-x))),
                lambda x: (1/(1+torch.exp(-x))) * (1 - (1/(1+torch.exp(-x))))),
    }

def ELU(x):
    return torch.where(x > 0, x, alpha * torch.exp(x) - 1)

def ELUDerivative(x):
    return torch.where(x > 0, torch.ones_like(x), alpha * torch.exp(x))

def normalise(x):
    return x/16.0

def startTraining():
    hiddenNodesWeights = torch.rand(inputNodes,hiddenNodes) * math.sqrt(2/inputNodes)
    outputNodesWeights = torch.rand(hiddenNodes,outputNodes) * math.sqrt(2/hiddenNodes)
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
    testingLoader = DataLoader(testingDataset, batch_size=batchSize, shuffle=False)

    return trainingLoader, testingLoader

#Forward propagation definitions
def forwardPropagation(inputs, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName):
    #matrix multiplication of hidden layer
    hiddenInputs = torch.matmul(inputs, hiddenWeights) + hiddenBias

    #Activation function applications
    tempActivationFunction, tempDerivativeFunction = activationFunctions[activationName]
    hiddenOutputs = tempActivationFunction(hiddenInputs)

    #matrix multiplication of output layer
    outputInputs = torch.matmul(hiddenOutputs, outputWeights) + outputBias

    return outputInputs, hiddenOutputs

#Training loop definitions
def trainingLoop(loader, hiddenWeights, outputWeights, hiddenBias, outputBias, learningRate, activationName):
    losses = []
    accuracies = []

    #an epoch is 1 complete pass of data
    for epoch in range(epochs):
        totalLoss = 0
        correctPredictions = 0
        totalSamples = 0   

        #Loop through batches with forward pass
        for batchIdx, (data, targets) in enumerate(loader):
            #Forward pass
            outputs, hiddenActivations = forwardPropagation(data, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)
            
            #loss calcs
            batchLoss = loss(outputs, targets)
            totalLoss += batchLoss.item()

            #Calculate accuracy for this batch
            predictions = torch.argmax(outputs, dim=1)
            correctPredictions += (predictions == targets).sum().item()
            totalSamples += targets.size(0)

            #backward pass
            hiddenWeightsGradient, outputWeightsGradient, hiddenBiasGradient, outputBiasGradient = backwardsPropagation(data, hiddenActivations, outputs, targets, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)

            # Update weights
            hiddenWeights, outputWeights, hiddenBias, outputBias = updateWeights(hiddenWeights, outputWeights, hiddenBias, outputBias,hiddenWeightsGradient, outputWeightsGradient,hiddenBiasGradient, outputBiasGradient)
        
        # Epoch statistics
        avgLoss = totalLoss / len(loader)
        accuracy = correctPredictions / totalSamples
        losses.append(avgLoss)
        accuracies.append(accuracy)
            
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avgLoss:.4f}, Accuracy: {accuracy:.4f}')

    return hiddenWeights, outputWeights, hiddenBias, outputBias, losses, accuracies


#Loss calculations
def loss(outputs, targets):
        return F.cross_entropy(outputs, targets)

#Backward propagation definitions (WHY IS THIS SO HARD?? MAYBE BCS I CANT ACTUALLY READ THE TENSORS?!?!!)
def backwardsPropagation(inputs, hiddenOutputs, outputs, targets, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName):
    #Converts output values into a normalised so the sum of the values = 1
    targetSoftmax = F.softmax(outputs, dim=1)
    
    #creates array full of 0's and 1 where the largest value is (this is the systems answer)
    targetOnehot = F.one_hot(targets, num_classes=outputNodes)

    outputError = (targetSoftmax - targetOnehot.float()) / batchSize                                    #dLoss/dOutputs (how wrong the answer is)

    #Gradients calculation (how much the weight and bias are wrong and need to be adjusted by)
    outputWeightsGradient = torch.matmul(hiddenOutputs.t(), outputError)                                #dLoss/dWeightOutputs (how much of the wrongness is from weight)
    outputBiasGradient = torch.sum(outputError, dim=0)                                                  #dLoss/dBiasOutputs (how much of the wrongness is from bias)
    #chain rule to bring error in output layer back to hidden layer
    hiddenError = torch.matmul(outputError, outputWeights.t())                                          #dLoss/dOutputs * dOutputs/dHiddenOutputs (moves error to hidden layer)

    #Now apply ELU derivative
    hiddenInputs = torch.matmul(inputs, hiddenWeights) + hiddenBias                                     #dLoss/dHiddenOutputs * dHiddenOutputs/dHiddenInputs
    tempActivationFunction, tempDerivativeFunction = activationFunctions[activationName]
    hiddenError *= tempDerivativeFunction(hiddenInputs)                                                          #Reverse ELU 

    hiddenWeightsGradient = torch.matmul(inputs.t(), hiddenError)                                       #dLoss/dWeightHidden (how much of the wrongness is from Weight)
    hiddenBiasGradient = torch.sum(hiddenError, dim=0)                                                  #dLoss/dBiasHidden (how much of the wrongness is from bias)

    return hiddenWeightsGradient, outputWeightsGradient, hiddenBiasGradient, outputBiasGradient

#updates weight function to apply gradients to current loops weights and biases
def updateWeights(hiddenWeights, outputWeights, hiddenBias, outputBias, hiddenWeightGradient, outputWeightsGradient, hiddenBiasGradient, outputBiasGradient):
    hiddenWeights -= learningRate * hiddenWeightGradient
    outputWeights -= learningRate * outputWeightsGradient
    hiddenBias -= learningRate * hiddenBiasGradient
    outputBias -= learningRate * outputBiasGradient

    return hiddenWeights, outputWeights, hiddenBias, outputBias

#Function def for data analysis
def evaluateModel(loader, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName):
    correctPredictions = 0
    totalPredictions = 0

    for data, targets in loader:
        outputs, _ = forwardPropagation(data, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)
        predictions = torch.argmax(outputs, dim=1)
        correctPredictions += (predictions == targets).sum().item()
        totalPredictions += targets.size(0)

    accuracy = correctPredictions/totalPredictions
    print(f'Test accuracy: {accuracy:.20f}')
    return accuracy

def analyseTestSet(loader, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName):
    correct = 0
    total = 0
    wrong_indices = []
    wrong_predictions = []
    wrong_targets = []
    
    for batch_idx, (data, targets) in enumerate(loader):
        outputs, _ = forwardPropagation(data, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)
        predictions = torch.argmax(outputs, dim=1)
        
        for i in range(len(targets)):
            if predictions[i] != targets[i]:
                wrong_indices.append(total + i)
                wrong_predictions.append(predictions[i].item())
                wrong_targets.append(targets[i].item())
        
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        
    
    print(f"Total test samples: {total}")
    print(f"Wrong predictions: {len(wrong_indices)}")
    print(f"Wrong indices: {wrong_indices}")
    print(f"Wrong predictions vs targets: {list(zip(wrong_predictions, wrong_targets))}")
    
    # Check if it's always the same samples
    if len(wrong_indices) > 0:
        print(f"Consistent error pattern: {len(set(wrong_indices)) == len(wrong_indices)}")

def plotResults(all_results):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 3, 1)
    colors = {'ELU': 'blue', 'ReLu': 'orange', 'Sigmoid': 'green'}
    
    for activation_name, results_dict in all_results.items():
        plt.plot(results_dict['losses'], label=activation_name, linewidth=2, color=colors.get(activation_name, 'black'))
    
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    plt.subplot(1, 3, 2)
    for activation_name, results_dict in all_results.items():
        plt.plot(results_dict['accuracies'], label=activation_name, linewidth=2, color=colors.get(activation_name, 'black'))
    
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)  # Set consistent y-axis for accuracy
    
    # Plot 3: Final Test Accuracy
    plt.subplot(1, 3, 3)
    test_accuracies = [results_dict['testAccuracies'] for results_dict in all_results.values()]
    activation_names = list(all_results.keys())
    
    # Use consistent colors
    bar_colors = [colors.get(name, 'gray') for name in activation_names]
    bars = plt.bar(activation_names, test_accuracies, color=bar_colors)
    
    plt.title('Final Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1.0)  # Zoom in to see differences better
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print a summary table for clarity
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for activation_name, results_dict in all_results.items():
        final_train_acc = results_dict['accuracies'][-1] if results_dict['accuracies'] else 0
        final_test_acc = results_dict['testAccuracies']
        print(f"{activation_name:8} | Train Acc: {final_train_acc:.4f} | Test Acc: {final_test_acc:.4f} | Wrong Predictions: {360 - int(final_test_acc * 360)}")

def exportWeights(hiddenWeight, outputWeights, hiddenBias, outputBias, activationName, filename=""):
    
    weights = {
        'activationFunction': activationName,
        'nodeArchitecture': f"{inputNodes}-{hiddenNodes}-{outputNodes}",
        'hiddenWeights': hiddenWeight.tolist(),
        'outputWeights': outputWeights.tolist(),
        'hiddenBias': hiddenBias.tolist(),
        'outputBias': outputBias.tolist(),
    }

    with open(f"{filename}Weights_{activationName}.json","w") as f:
        json.dump(weights, f, indent=2)

    print(f"Weights Exported")

#Main
def main():
    # Access the data and labels
    x = digits.data
    y = digits.target

    results = {}

    #Function for splitting data set up and holding some data points in reserve for data validation  and credibility during report writing
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

    trainingTensorsx, trainingTensorsy = dataPreprocessing(trainX, trainY)
    testingTensorsx, testingTensorsy = dataPreprocessing(testX, testY)

    for activationName in ['ELU', 'ReLu', 'Sigmoid']:
        print(f"\n============================================================")
        print(f"Training with {activationName} activation function")
        print(f"============================================================")

        trainingLoader, testLoader = dataloading(trainingTensorsx, trainingTensorsy, testingTensorsx, testingTensorsy)

        hiddenWeights, outputWeights, hiddenBias, outputBias = startTraining()

        hiddenWeights, outputWeights, hiddenBias, outputBias, losses, accuracies = trainingLoop(trainingLoader, hiddenWeights, outputWeights, hiddenBias, outputBias, learningRate, activationName)

        analyseTestSet(testLoader, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)

        testAccuracies = evaluateModel(testLoader, hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)

        results[activationName] = {
            'losses': losses,
            'accuracies': accuracies,
            'testAccuracies': testAccuracies
        }

        exportWeights(hiddenWeights, outputWeights, hiddenBias, outputBias, activationName)

    plotResults(results)

if __name__ == "__main__":
    main()