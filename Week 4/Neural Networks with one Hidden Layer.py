"""

Neural Networks with one Hidden Layer

"""

# In[1]: Imports

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
import numpy as np
torch.manual_seed(0)

# In[2]
#Plotting function
def plot_decision_regions_2class(model, dataset):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = dataset.x.numpy()
    y = dataset.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1 , X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1 , X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()


# In[3]
#Accuracy function
def accuracy(model, dataset):
    return np.mean(dataset.y.view(-1).numpy() == (model(dataset.x)[:, 0] > 0.5).numpy())

# In[4]
#Class Neural Net with 1 hidden layer
class SimpleNet(nn.Module):
    #Constructor
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H) #input layer
        self.linear2 = nn.Linear(H, D_out) #Output layer
        
    #Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        
        return x
        
# In[5]
#Training function
def train_model(data_set, model, criterion, train_loader, optimizer, epochs = 5):
    COST = []
    ACC = []
    for epoch in range(epochs):
        total=0
        for x, y in train_loader:
            
            optimizer.zero_grad()
            
            yhat = model(x)
            
            loss = criterion(yhat, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            #cumulative loss 
            total+=loss.item()
        ACC.append(accuracy(model, data_set))
        COST.append(total)
   
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(COST, color = color)
    ax1.set_xlabel('epoch', color = color)
    ax1.set_ylabel('total loss', color = color)
    ax1.tick_params(axis = 'y', color = color)
     
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
    ax2.plot(ACC, color=color)
    ax2.tick_params(axis='y', color = color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.show()

    return COST    
    
# In[6]
#Data Class

class XOR_Data(Dataset):
    
    #Constructor
    def __init__(self, N_s = 100):
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        for i in range(N_s // 4):
            self.x[i, :] = torch.Tensor([0.0, 0.0]) 
            self.y[i, 0] = torch.Tensor([0.0])

            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([1.0])
    
            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([1.0])
    
            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([0.0])

            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s

    #Get
    def __getitem__(self, index):    
        return self.x[index],self.y[index]
    
    #Length
    def __len__(self):
        return self.len
    
    #Plot the data
    def plot_stuff(self):
        plt.plot(self.x[self.y[:, 0] == 0, 0].numpy(), self.x[self.y[:, 0] == 0, 1].numpy(), 'o', label="y=0")
        plt.plot(self.x[self.y[:, 0] == 1, 0].numpy(), self.x[self.y[:, 0] == 1, 1].numpy(), 'ro', label="y=1")
        plt.legend()
        

# In[7]
#Create XOR_Data object
dataset = XOR_Data()
dataset.plot_stuff()

# In[8]
#Creating model and train

model = SimpleNet(2,1,1) #Model with 2 inputs, 1 neuron and 1 output

#Train the model
learning_rate = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
train_loader = DataLoader(dataset = dataset, batch_size = 1)
LOSS12 = train_model(dataset, model, criterion, train_loader, optimizer, epochs = 500)
plot_decision_regions_2class(model, dataset) 

model = SimpleNet(2,2,1) #Model with 2 inputs, 2 neuron and 1 output
#Train the model
learning_rate = 0.1
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
train_loader = DataLoader(dataset = dataset, batch_size = 1)
LOSS12 = train_model(dataset, model, criterion, train_loader, optimizer, epochs = 500)
plot_decision_regions_2class(model, dataset) 


model = SimpleNet(2,6,1) #Model with 2 inputs, 6 neuron and 1 output
#Train the model
learning_rate = 0.1
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
train_loader = DataLoader(dataset = dataset, batch_size = 1)
LOSS12 = train_model(dataset, model, criterion, train_loader, optimizer, epochs = 500)
plot_decision_regions_2class(model, dataset) 
































       