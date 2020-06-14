"""

Simple One Hidden Layer Neural Network

"""

# In[1] Imports

import torch 
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
import numpy as np
from torch import optim
torch.manual_seed(0)


# In[2]
    
#Function for plotting the model
def PlotStuff(X, Y, model, epoch, leg=True):
    
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg == True:
        plt.legend()
    else:
        pass

    
# In[3]

#Neutral Network Module Class and Training Function
class SimpleNet(nn.Module):
    
    #Constructor
    def __init__(self, D_in, H, D_out):
        super().__init__()
        
        #Hidden layer
        self.linear1 = nn.Linear(D_in, H) #Input dimension, Number of neurons.
        self.linear2 = nn.Linear(H, D_out) # number of neurons H and output dimension
        
        #Define the first linear layer as an attribute. This is not good practice
        self.a1 = None
        self.l1 = None
        self.l2 = None
        
    #Prediction forward function
    def forward(self, x):
        
        self.l1 = self.linear1(x)
        self.a1 = sigmoid(self.l1)
        self.l2 = self.linear2(self.a1)
        
        yhat = sigmoid(self.linear2(self.a1))
       
        return yhat
        


# In[4]
    
#Define the training function
def train_model(X, Y, model, optimizer, criterion, epochs = 1000):
    
    cost = [] #Empty cost list to accumulate the cost
    total = 0
    
    for epoch in range(epochs):
        total = 0
        
        for x, y in zip(X, Y): 
            
            yhat = model(x)
           
            loss = criterion(yhat, y)
            
            loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()
            
            #Cumulative loss
            total += loss.item()
            
        cost.append(total)    
        
        if epoch % 500 == 0:
            
            PlotStuff(X, Y, model, epoch, leg = True)
            plt.show()
            model(X)
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('activations')
            plt.show()
        
    return cost


# In[5]
    
#Create Data
X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)

Y = torch.zeros(X.shape[0])

Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0


# In[6]
#Criterion function Cross-Entropy, Optimizer
def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out


#Initialize the model
model = SimpleNet(1, 2, 1)

#Learning rate 
learning_rate = 0.1

#Optimizer 
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# In[7]

#Training

cost_cross = train_model(X, Y, model, optimizer, criterion_cross, epochs = 1000)

plt.plot(cost_cross)
plt.xlabel('epoch')
plt.title('cross entropy loss')

# In[8]
#Prediction
x = torch.tensor([0.0])
yhat = model(x)
yhat

X_=torch.tensor([[0.0],[2.0],[3.0]])
Yhat=model(X_)
Yhat

Yhat=Yhat>0.5
Yhat

# In[9]

#Trying a model with MSE loss function. #Not good

criterion_mse = nn.MSELoss()

model = SimpleNet(1, 2, 1)

cost_mse = train_model(X, Y, model, optimizer, criterion_mse, epochs = 1000)

plt.plot(cost_mse)
plt.xlabel('epoch')
plt.title('MSE loss ')


















        

        