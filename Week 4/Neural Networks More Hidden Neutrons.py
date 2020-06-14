"""

Neural Networks More Hidden Neutrons

"""


# In[1] Imports

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)


# In[2]
    
#Functions for plotting the model
def get_hist(model, data_set):
    
    activations = model.activation(data_set.x)
    for i, activation in enumerate(activations):
       
        plt.hist(activation.numpy(), 4, density = True)
        
        plt.title("Activation layer " + str(i+1))
        
        plt.xlabel("Activation")
        
        plt.legend()
        
        plt.show()

def PlotStuff(X,Y,model = None, leg = False):
    
    plt.plot(X[Y==0].numpy(),Y[Y==0].numpy(),'or', label = 'training points y = 0 ' )
    plt.plot(X[Y==1].numpy(),Y[Y==1].numpy(),'ob', label = 'training points y = 1 ' )

    if model != None:
        plt.plot(X.numpy(),model(X).detach().numpy(), label = 'Neural Network ')

    plt.legend()
    plt.show()


    
# In[3]

#Neutral Network Module Class and Training Function
class SimpleNet(nn.Module):
    
    #Constructor
    def __init__(self, D_in, H, D_out):
        super().__init__()
        
        #Hidden layer
        self.linear1 = nn.Linear(D_in, H) #Input dimension, Number of neurons.
        self.linear2 = nn.Linear(H, D_out) # number of neurons H and output dimension
        
    #Prediction forward function
    def forward(self, x):
        
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
       
        return x
        


# In[4]
    
#Define the training function
def train_model(data_set, model, criterion, train_loader, optimizer, epochs = 10, plot_number = 10):
    
    cost = [] #Empty cost list to accumulate the cost
    
    for epoch in range(epochs):
        total = 0
        
        for x, y in train_loader: 
            
            yhat = model(x)
           
            loss = criterion(yhat, y)
            
            loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()
            
            #Cumulative loss
            total += loss.item()
            
        cost.append(total)    
        
        if epoch % plot_number == 0:
            PlotStuff(data_set.x, data_set.y, model)
    
    plt.figure()
    plt.plot(cost)
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()
            
    return cost


# In[5]
    
#Create Data Class
class Data(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20, 20, 100).view(-1,1)
  
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:,0]>-10)& (self.x[:,0]<-5)]=1
        self.y[(self.x[:,0]>5)& (self.x[:,0]<10)]=1
        self.y = self.y.view(-1,1)
        self.len = self.x.shape[0]
    def __getitem__(self,index):    
            
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

data_set = Data()
PlotStuff(data_set.x, data_set.y, leg = False)

# In[6]
#Criterion BCE Loss, Optimizer Adam, train_loader
criterion = nn.BCELoss()

#Initialize the model with 9 hidden layer neurons.
model = SimpleNet(1, 9, 1)

#Learning rate 
learning_rate = 0.1

#train_loader
train_loader = DataLoader(dataset = data_set, batch_size = 100)

#Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# In[7]

#Training

COST = train_model(data_set, model, criterion, train_loader, optimizer, epochs = 1000, plot_number = 200)

plt.plot(COST)
plt.xlabel('epoch')
plt.title('BCE loss')


