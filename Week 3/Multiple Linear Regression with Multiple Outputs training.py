"""

Multiple Linear Regression with Multiple Outputs training


"""

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader


#Set random seed to 1
torch.manual_seed(1)

#Create Data class from torch Dataset import
class Data2D(Dataset):
    
    #Constructor
    def __init__(self):
        self.x = torch.zeros(20, 2) #Returns a tensor filled with the scalar value 0 with the shape 20. (20 samples, 2D)
        self.x[:, 0] = torch.arange(-1, 1, 0.1) #fill first column of x with values interval -1 to 1 with 0.1 step
        self.x[:, 1] = torch.arange(-1, 1, 0.1) #fill second column of x with values interval -1 to 1 with 0.1 step
        self.w = torch.tensor([ [1.0,-1.0],[1.0,3.0]]) #Because we have 2 outputs, we need a second set of w parameters
        self.b = torch.tensor([[1.0,-1.0]]) #Because we have 2 outputs, we need a second b parameter
        self.f = torch.mm(self.x, self.w) + self.b    
        self.y = self.f + 0.01 * torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]

    #Getter
    def __getitem__(self, index):          
        return self.x[index], self.y[index]
    
    #Get Length
    def __len__(self):
        return self.len
    
#Create a Data2D object data_set
data_set = Data2D()

#Create a customized linear regression class from nn.Module class
class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    # Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat    
    
#Create linear regression model with 2 inputs and 2 outputs
model = linear_regression(2,2)    
print("The initialized model parameters: ", list(model.parameters()))    #Print model parameters in a list    


#Create the optimizer and give it model parameters and learning rate
optimizer = optim.SGD(model.parameters(), lr = 0.01)

#Create loss function for torch.nn
criterion = nn.MSELoss()    

#Create the dataloader using DataLoader and the data_set created and batch size 2
train_loader = DataLoader(dataset = data_set, batch_size = 1)
    

LOSS = [] #Empty list to store the error

#number of epochs
epochs = 100  

def train_model(epochs):
    for epochs in range(epochs):
    
        for x,y in train_loader:
        
            #make a prediction 
            yhat=model(x)
            
            #calculate the loss
            loss=criterion(yhat,y)
            
            #store loss/cost in LOSS list
            LOSS.append(loss.item())
            
            #clear gradient 
            optimizer.zero_grad()
            
            #Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            #Update parameters using gradient and learning rate
            optimizer.step()
  
#Train the model
train_model(100)           

plt.plot(LOSS)
plt.xlabel("iterations ")
plt.ylabel("Cost/total loss ")
plt.show()







































