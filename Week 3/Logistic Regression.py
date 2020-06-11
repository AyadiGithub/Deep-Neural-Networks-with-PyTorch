"""

Logistic Regression 

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)

#Create a tensor ranging from -100 to 100
z = torch.arange(-100,100,0.1).view(-1,1)
print("The tensor: ", z)

#create a sigmoid object from torch.nn
sig = nn.Sigmoid()

#Calculate yhat using sigmoid object
yhat = sig(z)
print("The prediction yhat: ", yhat)

#Plot results
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

#Lets try the sigmoid model in torch
yhat1 = torch.sigmoid(z) #Element-wise Sigmoid from torch

#Plot results
plt.plot(z.numpy(), yhat1.numpy())
plt.xlabel('z')
plt.ylabel('yhat1')

#The plots are identical. 

"""
Build a logistic Regression object using nn.sequential with 1D input
"""

#Create x and X tensor
x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)


#We call nn.Sequential with Linear first then sigmoid
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())


#Print the parameters
print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

#Predict x using model created
yhat = model(x)
print("The prediction: ", yhat)


#Lets predict X
yhat1 = model(X)
print("The prediction: ", yhat1)



"""
Custom Logistic Regression Class using nn.Module
"""

class logistic_regression(nn.Module):
    
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
        
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        
        return yhat

#Create x and X tensor with 1 input
x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)


model = logistic_regression(1) #Initializing model that will predict 1 dimension

#Lets view the randomly intialized parameters
print("list(model.parameters()):\n", list(model.parameters()))
print("\nmodel.state_dict():\n", model.state_dict())

#Predict x and X
yhat = model(x)
yhat1 = model(X)


#Lets create a new model that takes 2 inputs
model = logistic_regression(2)

#Create x and X tensor with 2 inputs
x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[-100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)

#Predict x and X with 2 inputs
yhat = model(x)
print(yhat)

yhat1 = model(X)
print(yhat1)





