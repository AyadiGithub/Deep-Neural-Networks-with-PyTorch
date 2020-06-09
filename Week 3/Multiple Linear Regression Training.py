"""

Multiple Linear Regression Training


"""

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader


#Set random seed to 1
torch.manual_seed(1)

#The function for plotting 2D
def Plot_2D_Plane(model, dataset, n=0):
    w1 = model.state_dict()['linear.weight'].numpy()[0][0]
    w2 = model.state_dict()['linear.weight'].numpy()[0][1]
    b = model.state_dict()['linear.bias'].numpy()

    # Data
    x1 = data_set.x[:, 0].view(-1, 1).numpy()
    x2 = data_set.x[:, 1].view(-1, 1).numpy()
    y = data_set.y.numpy()

    # Make plane
    X, Y = np.meshgrid(np.arange(x1.min(), x1.max(), 0.05), np.arange(x2.min(), x2.max(), 0.05))
    yhat = w1 * X + w2 * Y + b

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(x1[:, 0], x2[:, 0], y[:, 0],'ro', label='y') # Scatter plot
    
    ax.plot_surface(X, Y, yhat) # Plane plot
    
    ax.set_xlabel('x1 ')
    ax.set_ylabel('x2 ')
    ax.set_zlabel('y')
    plt.title('estimated plane iteration:' + str(n))
    ax.legend()

    plt.show()


#Create Data class from torch Dataset import
class Data2D(Dataset):
    
    #Constructor
    def __init__(self):
        self.x = torch.zeros(20, 2) #Returns a tensor filled with the scalar value 0 with the shape 20. (20 samples, 2D)
        self.x[:, 0] = torch.arange(-1, 1, 0.1) #fill first column of x with values interval -1 to 1 with 0.1 step
        self.x[:, 1] = torch.arange(-1, 1, 0.1) #fill second column of x with values interval -1 to 1 with 0.1 step
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b    
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0],1))
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
    
#Create linear regression model
model = linear_regression(2,1)    
print("The initialized model parameters: ", list(model.parameters()))    #Print model parameters in a list

#Create optimizer 
optimizer = optim.SGD(model.parameters(), lr = 0.1)    #Create optimizer with SGD and learning rate 0.1 with model intialized parameters

#Create loss function for torch.nn
criterion = nn.MSELoss()    

#Create the dataloader using DataLoader and the data_set created and batch size 2
train_loader = DataLoader(dataset = data_set, batch_size = 2)
    

LOSS = [] #Empty list to store the error

print("Before Training: ")
Plot_2D_Plane(model, data_set)   #The fit Plane with auto initialized parameters before training

epochs = 100  

#Create train model 
def train_model(epochs):    
    
    for epoch in range(epochs):
        
        for x,y in train_loader:
            
            yhat = model(x)
            
            loss = criterion(yhat, y)
            
            LOSS.append(loss.item())
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()     

#Train model for specified number of epochs.             
train_model(epochs)

print("After Training: ")
Plot_2D_Plane(model, data_set, epochs)  #Fit Plane after training   
    
    
    
#Lets visualize the Loss for each epoch
plt.plot(LOSS)
plt.xlabel("Iterations ")
plt.ylabel("Cost/total loss ")    
    
    
    
"""

Lets try a new model with different hyper parameters
    
"""    
    
#Create linear regression model
model1 = linear_regression(2,1)    
print("The initialized model parameters: ", list(model.parameters()))    #Print model parameters in a list

#Create optimizer 
optimizer = optim.SGD(model1.parameters(), lr = 0.1)    #Create optimizer with SGD and learning rate 0.1 with model intialized parameters

#Create the dataloader using DataLoader and the data_set created and batch size 2
train_loader = DataLoader(dataset = data_set, batch_size = 30)
  
LOSS1 = [] #Empty list to store the error

print("Before Training: ")
Plot_2D_Plane(model1, data_set)   #The fit Plane with auto initialized parameters before training

epochs = 100  

#Create train model 
def train_model1(epochs):    
    
    for epoch in range(epochs):
        
        for x,y in train_loader:
            
            yhat = model1(x)
            
            loss = criterion(yhat, y)
            
            LOSS1.append(loss.item())
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()     

#Train model for specified number of epochs.             
train_model1(epochs)

print("After Training: ")
Plot_2D_Plane(model1, data_set, epochs)  #Fit Plane after training   
    
    
    
#Lets visualize the Loss for each epoch
plt.plot(LOSS1)
plt.xlabel("Iterations ")
plt.ylabel("Cost/total loss ")        
    
    
#Lets create a validation dataset to calculate the total loss or cost for both models   
torch.manual_seed(2)     
val_data = Data2D()    
Y = val_data.y    
X = val_data.x   

print("total loss or cost for model: ",criterion(model(X),Y))
print("total loss or cost for model: ",criterion(model1(X),Y))    
