## Imports
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

## Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

## Googly drive access (for saving results)
from google.colab import drive
drive.mount("/content/drive")

RESULTS_PATH = "/content/drive/MyDrive/cifar_energy_results.json"

#
# Training and evaluation helpers (train for one epoch)
#

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train() #training mode

    #initialise tracking variables
    running_loss = 0.0
    correct = 0
    total = 0

    #loop through batches
    for inputs, targets in dataloader:
        #move data to GPU/CPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad() #Clear prev batch

        #Case of AMP
        if use_amp:
            with torch.cuda.amp.autocast(): #dedicate mixed-precision within block
                outputs = model(inputs) #call forward to get logits
                loss = criterion(outputs, targets) #calculate loss
            scaler.scale(loss).backward() #compute loss gradients
            scaler.step(optimizer) #adjust weights/stabilise
            scaler.update()

        #Case of non-AMP
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        """
        Update running loss,
        logits -> predicted class,
        Update running total samples,
        Count/update num of correct predictions
        """
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #Compute and return average loss per sample over batch
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

#
# Validation function
#
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval() #validation mode

    #initialise tracking variables
    running_loss = 0.0
    correct = 0
    total = 0

    #loop through batches
    for inputs, targets in dataloader:
        #move data to GPU/CPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        #call forward to get logits and calc loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #update variables
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #Compute avg loss over batch
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc
