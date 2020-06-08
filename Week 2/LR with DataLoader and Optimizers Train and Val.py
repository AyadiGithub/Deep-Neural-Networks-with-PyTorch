"""

Linear Regression with DataLoader, Pytorch way

"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.data import Dataset, DataLoader
from torch import optim

#Creating Data Class
class Data(Dataset):
    
    #Constructor
    def __init__(self, train = True):
            self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
            self.f = -3 * self.x + 1
            self.y = self.f + 0.1 * torch.randn(self.x.size())
            self.len = self.x.shape[0]
            
            #Creating outliers 
            if train == True:
                self.y[0] = 0
                self.y[50:55] = 20
            else:
                pass
      
    #Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index]
    
    #Get Length
    def __len__(self):
        return self.len

#Creating random data
torch.manual_seed(1)

#Creating train_data object and validation data
train_data = Data() 
val_data = Data(train = False) 

#Plot out training points
plt.plot(train_data.x.numpy(), train_data.y.numpy(), 'xr',label="training data ")
plt.plot(train_data.x.numpy(), train_data.f.numpy(),label="true function  ")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


class linear_regression(nn.Module): #Creating linear_regression class with attributes from nn.Module 

    #Constructor
    def __init__(self, input_size, output_size):
        super().__init__() #Inheriting methods from parent class nn.module
        self.linear = nn.Linear(input_size, output_size)
        
    #Prediction
    def forward(self, x):
        yhat = self.linear(x)
        return yhat
    

#Using Pytorch built-in functions to create a criterion function
#Using the MSE loss
criterion = nn.MSELoss()

#Create a DataLoader object
trainloader = DataLoader(dataset = train_data, batch_size = 1) #batch_size 1


# Create Learning Rate list, the error lists and the MODELS list
learning_rates=[0.0001, 0.001, 0.01, 0.1]
train_error=torch.zeros(len(learning_rates))
validation_error=torch.zeros(len(learning_rates))
MODELS=[]


# Define the train model function and train the model
def train_model_with_lr (iter, lr_list):
    
    #iterate through different learning rates 
    for i, lr in enumerate(lr_list):
        
        model = linear_regression(1, 1) #LR with 1 input and 1 output
        
        optimizer = optim.SGD(model.parameters(), lr = lr) #SGD Optimizer 
        
        for epoch in range(iter):
            
            for x, y in trainloader:
                yhat = model(x) #prediction
                
                loss = criterion(yhat, y) #loss calculation
                
                optimizer.zero_grad() #Zeroing gradient
                
                loss.backward() #backward pass
                
                optimizer.step() #Re-initialization
                
                print(model.state_dict()) #Printing the updated parameters for each iteration
                
        #Training Data
        Yhat = model(train_data.x)
        
        train_loss = criterion(Yhat, train_data.y)
        
        train_error[i] = train_loss.item()
    
        #Validation Data
        Yhat = model(val_data.x)
        
        val_loss = criterion(Yhat, val_data.y)
        
        validation_error[i] = val_loss.item()
        
        MODELS.append(model)

train_model_with_lr(10, learning_rates)



#Plot the training loss and validation loss
#Validation error will be smaller because Outliers were added to the train_data for visualization 
#Plotting log plot due to learning being in order of 10
plt.semilogx(np.array(learning_rates), train_error.numpy(), label = 'training loss/total Loss') 
plt.semilogx(np.array(learning_rates), validation_error.numpy(), label = 'validation cost/total Loss')
plt.ylabel('Cost\ Total Loss')
plt.xlabel('learning rate')
plt.legend()
plt.show()



#Plot the predictions
i = 0
for model, learning_rate in zip(MODELS, learning_rates):
    
    yhat = model(val_data.x) #Making predictions on Val_data
    
    plt.plot(val_data.x.numpy(), yhat.detach().numpy(), label = 'lr:' + str(learning_rate)) #Plot yhat for each val_data.x
    
    print('i', yhat.detach().numpy()[0:3])
    
#Plotting learning rate versus validation data. 
plt.plot(val_data.x.numpy(), val_data.f.numpy(), 'or', label = 'validation data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
























