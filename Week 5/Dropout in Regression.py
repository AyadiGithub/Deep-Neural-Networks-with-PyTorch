"""

Comparing Neural Network Dropout vs Non in Regression

"""


# In[1] Imports

import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# In[2] Creating Data

class Data(Dataset):
    
    # Constructor
    def __init__(self, N_SAMPLES=40, noise_std=1, train=True):
        self.x = torch.linspace(-10, 10, N_SAMPLES).view(-1, 1)
        self.f = self.x ** 3
        if train != True:
            torch.manual_seed(1)
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
            torch.manual_seed(0)
        else:
            self.y = self.f + noise_std * torch.randn(self.f.size())
            self.y = self.y.view(-1, 1)
            
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
    # Plot the data
    def plot(self):
        plt.figure(figsize = (9, 14))
        plt.scatter(self.x.numpy(), self.y.numpy(), label="Samples")
        plt.plot(self.x.numpy(), self.f.numpy() ,label="True Function", color='orange')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((-10, 10))
        plt.ylim((-20, 20))
        plt.legend(loc="best")
        plt.show()

# In[3] Creating Network 


class Network(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, H3, H4, D_out, p=0): # dropout disabled by default p=0
        super().__init__()
        self.drop = nn.Dropout(p=p) # Dropout with p 
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, H4)
        self.linear5 = nn.Linear(H4, D_out)
        
    # Prediction
    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear4(x)
        x = torch.relu(x)
        x = self.linear5(x)
        return x

        
# In[4] Training function


def train_model(epochs):
    
    for epoch in range(epochs):
        #all the samples are used for training 
        yhat = model(data_set.x)
        yhat_dropout = model_dropout(data_set.x)
        loss = criterion(yhat, data_set.y)
        loss_dropout = criterion(yhat_dropout, data_set.y)

        #store the loss for both the training and validation data for both models 
        LOSS['training data no dropout'].append(loss.item())
        LOSS['validation data no dropout'].append(criterion(model(validation_set.x), validation_set.y).item())
        LOSS['training data dropout'].append(loss_dropout.item())
        model_dropout.eval()
        LOSS['validation data dropout'].append(criterion(model_dropout(validation_set.x), validation_set.y).item())
        model_dropout.train()

        optimizer.zero_grad()
        optimizer_dropout.zero_grad()
        loss.backward()
        loss_dropout.backward()
        optimizer.step()
        optimizer_dropout.step()


# In[5] Create Dataset object and plot

data_set = Data(N_SAMPLES=1000, noise_std=10, train=True)
data_set.plot()
validation_set = Data(train=False)

# In[6] Create models with dropout and without, Create optimizer and criterion

model = Network(1, 250, 250, 250, 250,  1)
model_dropout = Network(1, 250, 250, 250, 250, 1, p=0.2) #dropout of probability 0.5

model_dropout.train() # Set model to train model. It is trian by default but good practice to set it anyway

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()


# In[7] Initialize the Loss dictionary to store Loss

LOSS = {}
LOSS['training data no dropout'] = []
LOSS['validation data no dropout'] = []
LOSS['training data dropout'] = []
LOSS['validation data dropout'] = []

# In[8] train models for 500 epochs and evaluate

train_model(10000)

# Set model_dropout to evaluation mode. This is done to disable dropout

model_dropout.eval()
yhat = model(data_set.x)
yhat_drop = model_dropout(data_set.x)


# In[9] Plot results to show decision model for model and model_dropout

# Plot predictions for model and model_dropout
plt.figure(figsize=(6.1, 10))
plt.scatter(data_set.x.numpy(), data_set.y.numpy(), label="Samples")
plt.plot(data_set.x.numpy(), data_set.f.numpy(), label="True function", color='orange')
plt.plot(data_set.x.numpy(), yhat.detach().numpy(), label='no dropout', c='r')
plt.plot(data_set.x.numpy(), yhat_drop.detach().numpy(), label="dropout", c ='g')
# Plot labels and limits
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((-10, 10))
plt.ylim((-20, 20))
plt.legend(loc = "best")
plt.show()

# LOSS Plot
# Plot the loss
plt.figure(figsize=(9, 14))
for key, value in LOSS.items():
    plt.plot(np.log(np.array(value)), label=key)
    plt.legend()
    plt.xlabel("iterations")
    plt.ylabel("Log of cost or total loss")

