import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split

# Load the dataset and train, val, test splits
print("Loading datasets...")
FASHION_transform = transforms.Compose([
    transforms.ToTensor(), # Transform from [0,255] uint8 to [0,1] float
    transforms.Normalize([0.2859], [0.3530]) # Normalize to zero mean and unit variance
])
FASHION_trainval = datasets.FashionMNIST('.', download=True, train=True, transform=FASHION_transform)
FASHION_train = Subset(FASHION_trainval, range(50000))
FASHION_val = Subset(FASHION_trainval, range(50000,60000))
FASHION_test = datasets.FashionMNIST('.', download=True, train=False, transform=FASHION_transform)
print("Done!")

# Create dataloaders
# TODO: Experiment with different batch sizes
trainloader = DataLoader(FASHION_train, batch_size=64, shuffle=True)
valloader = DataLoader(FASHION_val, batch_size=64, shuffle=True)
testloader = DataLoader(FASHION_test, batch_size=64, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Design your own network, define layers here.
        # Here We provide a sample of two-layer fully-connected network from HW4 Part3.
        # Your solution, however, should contain convolutional layers.
        # Refer to PyTorch documentations of torch.nn to pick your layers. (https://pytorch.org/docs/stable/nn.html)
        # Some common Choices are: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout
        # If you have many layers, consider using nn.Sequential() to simplify your code
        self.conv1 = nn.Conv2d(1,5,5,1, padding = 2)
        self.conv2 = nn.Conv2d(5,10,5,1, padding = 2)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(10*14*14, 1000) # from 28x28 input image to hidden layer of size 256
        self.fc3 = nn.Linear(1000, 80)
        self.fc2 = nn.Linear(80,10) # from hidden layer to 10 class scores

    def forward(self,x):
        relu = nn.ReLU()
        # TODO: Design your own network, implement forward pass here
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.pool(x)
        x = x.view(-1,10*14*14) # Flatten each image in the batch
        x = self.fc1(x)
        x = relu(x)
        x = self.fc3(x)
        x = relu(x)
        x = self.fc2(x)
        # The loss layer will be applied outside Network class
        return x

device = "cuda" if torch.cuda.is_available() else "cpu" # Configure device
print(device)
model = Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer

# TODO: Modify the line below, experiment with different optimizers and parameters (such as learning rate)
optimizer = optim.Adam(model.parameters(), lr=4e-3, weight_decay=1e-4) # Specify optimizer and assign trainable parameters to it, weight_decay is L2 regularization strength
num_epoch = 20 # TODO: Choose an appropriate number of training epochs

def train(model, loader, num_epoch = 10): # Train the model
    print("Start training...")
    model.train() # Set the model to training mode
    for i in range(num_epoch):
        running_loss = []
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad() # Clear gradients from the previous iteration
            pred = model(batch) # This will call Network.forward() that you implement
            loss = criterion(pred, label) # Calculate the loss
            running_loss.append(loss.item())
            loss.backward() # Backprop gradients to all tensors in the network
            optimizer.step() # Update trainable weights
        print("Epoch {} loss:{}".format(i+1,np.mean(running_loss))) # Print the average loss for this epoch
    print("Done!")

def evaluate(model, loader): # Evaluate accuracy on validation / test set
    model.eval() # Set the model to evaluation mode
    correct = 0
    with torch.no_grad(): # Do not calculate grident to speed up computation
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            pred = model(batch)
            correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("Evaluation accuracy: {}".format(acc))
    return acc
    
train(model, trainloader, num_epoch)
print("Evaluate on validation set...")
evaluate(model, valloader)
print("Evaluate on test set")
evaluate(model, testloader)
