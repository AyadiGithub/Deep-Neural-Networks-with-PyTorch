"""

Linear Regression with DataLoader, Pytorch way

"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.data import Dataset, DataLoader

#Add a class plot_error_surfaces to visualize the data space and parameters.
class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30, 30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1, cmap = 'viridis', edgecolor = 'none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
            
    # Setter
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.LOSS.append(loss)
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
        
    # Plot diagram    
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim()
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour Iteration' + str(self.n) )
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

#Creating random data
torch.manual_seed(1)


#Create Data class to create dataset objects
class Data(Dataset):
    
    #Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = 1 * self.x - 1
        self.y = self.f + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape[0]
        
    #Getter
    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    #Get Length
    def __len__(self):
        return self.len

#Creating Data object dataset
dataset = Data()



#Lets plot Y, X, f
plt.plot(dataset.x.numpy(), dataset.y.numpy(), 'rx', label = 'y')
plt.plot(dataset.x.numpy(), dataset.f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

"""

Creating the model for Linear Regression and Total Loss function (Cost)

"""

from torch import nn, optim #Importing nn class and optimizer

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

#Lets create a Linear regression object and optimizer object

model = linear_regression(1,1)
#We will use Stochastic Gradient Descent, SGD,  as the optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.01) #model.parameters() takes our model parameters created in dataset

#Lets check the parameters
list(model.parameters())

#Lets check the optimizer dictionary
optimizer.state_dict()

#Create a DataLoader object
trainloader = DataLoader(dataset = dataset, batch_size = 1)
#Pytorch automatically and randomly initializes parameters, as seen using model.state_dict()
model.state_dict()

#Lets specify the parameters to make the process longer and visualize the training
model.state_dict()['linear.weight'][0] = -15
model.state_dict()['linear.bias'][0] = -10

#Create plot surface object
get_surface = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go = True)


"""

Training the model using Batch Gradient Descent

"""

LOSS_BGD = []

#Train Model function
def train_model_BGD(iter):
    for epoch in range(iter):
        for x,y in trainloader:
            
            yhat = model(x) #Predict yhat using initialized parameters
            
            loss = criterion(yhat, y) #Calculate the loss MSE 
            
            get_surface.set_para_loss(model, loss.tolist()) #Plot  
            
            #store the loss in the list LOSS_BGD
            LOSS_BGD.append(loss) #Add the loss to the list LOSS_BGD
            
            optimizer.zero_grad() #Zeros the gradient because otherwise pytorch accumulates it
            
            loss.backward() 

            optimizer.step() #Updates parameters
            
            get_surface.plot_ps()
        
        
#Now we train the model
train_model_BGD(5)

#Lets see the parameters of the model after 5 epochs
model.state_dict()



"""

Lets try a different learning rate.

"""
#Initializing the new model1 with lr = 0.1
model1 = linear_regression(1,1)

#Manually Initializing the parameters 
model1.state_dict()['linear.weight'][0] = -15
model1.state_dict()['linear.bias'][0] = -10

#Initializing the plot
get_surface = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go = False)

#Create a DataLoader object
trainloader = DataLoader(dataset = dataset, batch_size = 1)

#Set the optimizer
optimizer = optim.SGD(model1.parameters(), lr = 0.1) #model.parameters() takes our model parameters created in dataset

LOSS_BGD1 = []

def train_model_BGD1(iter):
    for epoch in range(iter):
        for x,y in trainloader:
            
            yhat = model1(x) #Predict yhat using initialized parameters
            
            loss = criterion(yhat, y) #Calculate the loss MSE 
            
            get_surface.set_para_loss(model1, loss.tolist()) #Plot  
            
            #store the loss in the list LOSS_BGD
            LOSS_BGD1.append(loss) #Add the loss to the list LOSS_BGD
            
            optimizer.zero_grad() #Zeros the gradient because otherwise pytorch accumulates it
            
            loss.backward() 

            optimizer.step() #Updates parameters
            
            get_surface.plot_ps()

#Train model with lr = 0.1
train_model_BGD1(5)

#Lets see the parameters of the model after 5 epochs
model1.state_dict()

plt.plot(LOSS_BGD,label = " Batch Gradient Descent with lr 0.01")
plt.plot(LOSS_BGD1, label = " Batch Gradient Descent with lr 0.1", linestyle='dashed')
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()



"""

LETS TRY Adam optimizer

"""

#Initializing the new model1 with lr = 0.1
model2 = linear_regression(1,1)

#Manually Initializing the parameters 
model2.state_dict()['linear.weight'][0] = -15
model2.state_dict()['linear.bias'][0] = -10

#Initializing the plot
get_surface = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go = False)

#Create a DataLoader object
trainloader = DataLoader(dataset = dataset, batch_size = 1)

#Set the optimizer
optimizer = optim.Adam(model2.parameters(), lr = 0.1)  #model.parameters() takes our model parameters created in dataset

LOSS_ADAM = []

def train_model_ADAM(iter):
    for epoch in range(iter):
        for x,y in trainloader:
            
            yhat = model2(x) #Predict yhat using initialized parameters
            
            loss = criterion(yhat, y) #Calculate the loss MSE 
            
            get_surface.set_para_loss(model2, loss.tolist()) #Plot  
            
            #store the loss in the list LOSS_BGD
            LOSS_ADAM.append(loss) #Add the loss to the list LOSS_BGD
            
            optimizer.zero_grad() #Zeros the gradient because otherwise pytorch accumulates it
            
            loss.backward() 

            optimizer.step() #Updates parameters
            
            get_surface.plot_ps()

#Train model with lr = 0.1
train_model_ADAM(5)

#Lets compare the models
plt.plot(LOSS_BGD,label = " Batch Gradient Descent with lr 0.01")
plt.plot(LOSS_BGD1, label = " Batch Gradient Descent with lr 0.1", linestyle='dashed')
plt.plot(LOSS_ADAM, label = " Batch Gradient Descent with Adam", linestyle='dotted')
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()









