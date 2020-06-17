"""

Comparing Neural Network Dropout vs Non in Classification

"""


# In[1] Imports

import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset

# In[2] Plotting function


def plot_decision_regions_3class(data_set, model=None):
    cmap_light = ListedColormap([ '#0000FF','#FF0000'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    newdata = np.c_[xx.ravel(), yy.ravel()]
    
    Z = data_set.multi_dim_poly(newdata).flatten()
    f = np.zeros(Z.shape)
    f[Z > 0] = 1
    f = f.reshape(xx.shape)
    if model != None:
        model.eval()
        XX = torch.Tensor(newdata)
        _, yhat = torch.max(model(XX), 1)
        yhat = yhat.numpy().reshape(xx.shape)
        plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
    else:
        plt.contour(xx, yy, f, cmap=plt.cm.Paired)
        plt.pcolormesh(xx, yy, f, cmap=cmap_light) 

    plt.title("decision region vs True decision boundary")

# In[3] Creating Data

class Data(Dataset):
    
    # Constructor
    def __init__(self, N_SAMPLES=1000, noise_std=0.15, train=True):
        a = np.matrix([-1, 1, 2, 1, 1, -3, 1]).T
        self.x = np.matrix(np.random.rand(N_SAMPLES, 2))
        self.f = np.array(a[0] + (self.x) * a[1:3] + np.multiply(self.x[:, 0], self.x[:, 1]) * a[4] + np.multiply(self.x, self.x) * a[5:7]).flatten()
        self.a = a
       
        self.y = np.zeros(N_SAMPLES)
        self.y[self.f > 0] = 1
        self.y = torch.from_numpy(self.y).type(torch.LongTensor)
        self.x = torch.from_numpy(self.x).type(torch.FloatTensor)
        self.x = self.x + noise_std * torch.randn(self.x.size())
        self.f = torch.from_numpy(self.f)
        self.a = a
        if train == True:
            torch.manual_seed(1)
            self.x = self.x + noise_std * torch.randn(self.x.size())
            torch.manual_seed(0)
        
    # Getter        
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
    
    # Plot the diagram
    def plot(self):
        X = data_set.x.numpy()
        y = data_set.y.numpy()
        h = .02
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max() 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = data_set.multi_dim_poly(np.c_[xx.ravel(), yy.ravel()]).flatten()
        f = np.zeros(Z.shape)
        f[Z > 0] = 1
        f = f.reshape(xx.shape)
        
        plt.title('True decision boundary  and sample points with noise ')
        plt.plot(self.x[self.y == 0, 0].numpy(), self.x[self.y == 0,1].numpy(), 'bo', label='y=0') 
        plt.plot(self.x[self.y == 1, 0].numpy(), self.x[self.y == 1,1].numpy(), 'ro', label='y=1')
        plt.contour(xx, yy, f,cmap=plt.cm.Paired)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.legend()
    
    # Make a multidimension ploynomial function
    def multi_dim_poly(self, x):
        x = np.matrix(x)
        out = np.array(self.a[0] + (x) * self.a[1:3] + np.multiply(x[:, 0], x[:, 1]) * self.a[4] + np.multiply(x, x) * self.a[5:7])
        out = np.array(out)
        return out

# In[4] Creating Network 


class Network(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out, p=0): # dropout disabled by default p=0
        super().__init__()
        self.drop = nn.Dropout(p=p) # Dropout with p 
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
        
    # Prediction
    def forward(self, x):
        x = self.linear1(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x

        
# In[5] Training function


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


# In[6] accuracy function


def accuracy(model, data_set):
    _, yhat = torch.max(model(data_set.x), 1)
    return (yhat == data_set.y).numpy().mean()


# In[7] Create Dataset object and plot

data_set = Data(noise_std=0.2)
data_set.plot()
validation_set = Data(train=False)

# In[8] Create models with dropout and without, Create optimizer and criterion

model = Network(2, 300, 300, 2)
model_dropout = Network(2, 300, 300, 2, p=0.2) #dropout of probability 0.5

model_dropout.train() # Set model to train model. It is trian by default but good practice to set it anyway

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer_dropout = torch.optim.Adam(model_dropout.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


# In[9] Initialize the Loss dictionary to store Loss

LOSS = {}
LOSS['training data no dropout'] = []
LOSS['validation data no dropout'] = []
LOSS['training data dropout'] = []
LOSS['validation data dropout'] = []

# In[10] train models for 500 epochs and evaluate

train_model(500)

# Set model_dropout to evaluation mode. This is done to disable dropout

model_dropout.eval()

# Print out the accuracy of the model without dropout
print("The accuracy of the model without dropout: ", accuracy(model, validation_set))

# Print out the accuracy of the model with dropout
print("The accuracy of the model with dropout: ", accuracy(model_dropout, validation_set))


# In[11] Plot results to show decision model for model and model_dropout

# Plot the decision boundary and the prediction
plot_decision_regions_3class(data_set) # What the prediction should be

# The model without dropout
plot_decision_regions_3class(data_set, model)

# The model with dropout
plot_decision_regions_3class(data_set, model_dropout)


# LOSS Plot
plt.figure(figsize=(6.1, 10))
def plot_LOSS():
    for key, value in LOSS.items():
        plt.plot(np.log(np.array(value)), label=key)
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("Log of cost or total loss")

plot_LOSS()