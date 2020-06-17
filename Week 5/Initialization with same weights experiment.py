"""

Neural Network same weights Initialization Experiment

"""


# In[1] Imports

import torch 
import torch.nn as nn
import matplotlib.pylab as plt
torch.manual_seed(0)
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# In[2] Creating Data

# Create the train dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Create the validation dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create Dataloader for both train dataset and validation dataset

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)


# In[3] Creating Networks using ModuleList 

# Network with relu activation and He_initialization with relu
class Network_He(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# Network with relu activation and uniform initialization
class Network_Uniform(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size,output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)
    
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
                
        return x
  

# Network with relu and Default Initilization
class Network(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
        
    def forward(self, x):
        L=len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
                
        return x

# Network with relu and Same Initilization
class Network_Same(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)
        
    def forward(self, x):
        L=len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
                
        return x

# Network with tanh activation and Xavier Initialization for Tanh
class Network_Xavier(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)
    
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x


# In[5] Training function


def train(model, criterion, train_loader, validation_loader, optimizer, epochs = 100):
    i = 0
    loss_accuracy = {'training_loss':[], 'validation_accuracy':[]}  
    
    for epoch in range(epochs):
        for i,(x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_accuracy['training_loss'].append(loss.data.item())
            
        correct = 0
        for x, y in validation_loader:
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label==y).sum().item()
        accuracy = 100 * (correct / len(validation_dataset))
        loss_accuracy['validation_accuracy'].append(accuracy)
        
    return loss_accuracy


# In[4] Creating the layer for ModuleList

input_dim = 28*28
output_dim = 10
layers = [input_dim, 100, 10, 100, 10, 100, output_dim]
epochs = 10


# In[5] Creating loss function and Optimizers and Networks for comparison

criterion = nn.CrossEntropyLoss()

# First model with default initialization
model = Network(layers)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=epochs)

# 2nd model with Uniform Initialization
model_Uniform = Network_Uniform(layers)
optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=0.01)
training_results_Uniform = train(model_Uniform, criterion, train_loader, validation_loader, optimizer, epochs=epochs)


# 3rd model with Xavier_initialization
model_Xavier = Network_Xavier(layers)
optimizer = torch.optim.SGD(model_Xavier.parameters(), lr=0.01)
training_results_Xavier = train(model_Xavier, criterion, train_loader, validation_loader, optimizer, epochs=epochs)


# 4th model with He_initialization
model_He = Network_He(layers)
optimizer = torch.optim.SGD(model_He.parameters(), lr=0.01)
training_results_He = train(model_He, criterion, train_loader, validation_loader, optimizer, epochs=epochs)

# 5th model with the same initializations for all neurons
model_Same = Network_Same(layers)
model_Same.state_dict()['hidden.0.weight'][:]=1.0
model_Same.state_dict()['hidden.1.weight'][:]=1.0
model_Same.state_dict()['hidden.2.weight'][:]=1.0
model_Same.state_dict()['hidden.3.weight'][:]=1.0
model_Same.state_dict()['hidden.4.weight'][:]=1.0
model_Same.state_dict()['hidden.5.weight'][:]=1.0
model_Same.state_dict()['hidden.0.bias'][:]=0.0
model_Same.state_dict()['hidden.1.bias'][:]=0.0
model_Same.state_dict()['hidden.2.bias'][:]=0.0
model_Same.state_dict()['hidden.3.bias'][:]=0.0
model_Same.state_dict()['hidden.4.bias'][:]=0.0
model_Same.state_dict()['hidden.5.bias'][:]=0.0
model_Same.state_dict()

optimizer = torch.optim.SGD(model_Same.parameters(), lr=0.01)
training_results_Same = train(model_Same, criterion, train_loader, validation_loader, optimizer, epochs=epochs)



# In[5] Plotting results

# Plot the loss
plt.plot(training_results_Xavier['training_loss'], label='Xavier')
plt.plot(training_results_He['training_loss'], label='He')
plt.plot(training_results['training_loss'], label='Default')
plt.plot(training_results_Uniform['training_loss'], label='Uniform')
plt.plot(training_results_Same['training_loss'], label='Same')
plt.ylabel('loss')
plt.xlabel('iteration ')  
plt.title('training loss iterations')
plt.legend()


# Plot the accuracy
plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier')
plt.plot(training_results_He['validation_accuracy'], label='He')
plt.plot(training_results['validation_accuracy'], label='Default')
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform') 
plt.plot(training_results_Same['validation_accuracy'], label='Same') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs')   
plt.legend()

