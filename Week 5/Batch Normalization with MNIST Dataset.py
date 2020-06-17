"""

Batch Normalization with MNIST Dataset

"""


# In[1] Imports

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)


# In[2] Create Data/Load MNIST Dataset

# load the train dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# load the train dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create Data Loader for both train and validating
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


# In[3] Create Neural Network with BatchNorm and without

# NN with BatchNorm
class NetworkBatchNorm(nn.Module):
    
    # Constructor
    def __init__(self, in_size, H1, H2, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, out_size)
        self.bn1 = nn.BatchNorm1d(H1)
        self.bn2 = nn.BatchNorm1d(H2)

    # Prediction
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.linear3(x)
        return x
    
    # Activations to analyze the results
    def activation(self, x):
        out = []
        z1 = self.bn1(self.linear1(x))
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.relu(z1)
        out.append(a1.detach().numpy().reshape(-1))
        z2 = self.bn2(self.linear2(a1))
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.relu(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out
 
       
# NN without BatchNorm
class Network(nn.Module):
    
    # Constructor
    def __init__(self, in_size, H1, H2, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, out_size)

    # Prediction
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x
    
    # Activations to analyze the results
    def activation(self, x):
        out = []
        z1 = self.linear1(x)
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.relu(z1)
        out.append(a1.detach().numpy().reshape(-1))
        z2 = self.linear2(a1)
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.relu(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out


# In[4] Create train loop

# Define the function to train model


def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss':[], 'validation_accuracy':[]}  

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
            
        correct = 0
        for x, y in validation_loader:
            model.eval()
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()
            
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff
        

# In[6] Initialize the model, optimizer, criterion

# Loss function
criterion = nn.CrossEntropyLoss()

# Initializing both models
model_batchnorm = NetworkBatchNorm(28*28, 100, 100, 10)
model = Network(28*28, 100, 100, 10)

# Optimizer and train
optimizer = torch.optim.Adam(model_batchnorm.parameters(), lr=0.1)
training_results_Norm = train(model_batchnorm, criterion, train_loader, validation_loader, optimizer, epochs=5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=5)


# In[5] Model evaluation and Plotting

# set models to evaluation so that batchnorm is put in eval mode.
model.eval()
model_batchnorm.eval()

# Plot model activations
out = model.activation(validation_dataset[0][0].reshape(-1,28*28))
plt.hist(out[2], label = 'model with no batch normalization' )
plt.xlabel("activation ")
plt.legend()
plt.show()

out_batchnorm = model_batchnorm.activation(validation_dataset[0][0].reshape(-1,28*28))
plt.hist(out_batchnorm[2], label = 'model with normalization')
plt.xlabel("activation ")
plt.legend()
plt.show()


# Plot the diagram to show the loss
plt.plot(training_results['training_loss'], label = 'No Batch Normalization')
plt.plot(training_results_Norm['training_loss'], label = 'Batch Normalization')
plt.ylabel('Cost')
plt.xlabel('iterations ')   
plt.legend()
plt.show()


# Plot the diagram to show the accuracy
plt.plot(training_results['validation_accuracy'], label = 'No Batch Normalization')
plt.plot(training_results_Norm['validation_accuracy'], label = 'Batch Normalization')
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')   
plt.legend()
plt.show()
