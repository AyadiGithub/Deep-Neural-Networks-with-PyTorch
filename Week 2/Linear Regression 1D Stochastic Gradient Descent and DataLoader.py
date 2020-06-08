"""

Linear Regression 1D: Prediction Stochastic Gradient Descent (SGD) and the DataLoader
 
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

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
            plt.axes(projection = '3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
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
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
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
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label = "training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()


#Creating random data
torch.manual_seed(1)

#Generating values
X = torch.arange(-3, 3, 0.1).view(-1, 1) #X -3 to 3 with 0.1 steps
f = 1 * X - 1
Y = f + 0.1 * torch.randn(X.size()) #Adding random noise 

#Lets plot Y, X, f
plt.plot(X.numpy(), Y.numpy(), 'rx', label = 'y')
plt.plot(X.numpy(), f.numpy(), label = 'f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
 
# Define the forward function
def forward(x):
    return w * x + b

# Define the MSE Loss function
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2) #Mean Squared Error 

#We use plot_error_surfaces to visualize
get_surface = plot_error_surfaces(15, 13, X, Y, 30)

# Define the parameters w, b for y = wx + b
w = torch.tensor(-15.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.
b = torch.tensor(-10.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.


#Define a learning rate and create an empty list LOSS to store the loss for each iteration
lr = 0.1 #Our first choice learning rate
LOSS_BGD = []

#The functipn for training the model. We will use the functions we create in it. 
#We Will use Batch Gradient Descent
def train_model(iter):
    
    #Epoch loop
    for epoch in range(iter):
        # make a prediction
        Yhat = forward(X)
        
        # calculate the loss 
        loss = criterion(Yhat, Y)

        # Section for plotting
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        get_surface.plot_ps()
            
        # store the loss in the list LOSS_BGD
        LOSS_BGD.append(loss)
        
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # update parameters slope and bias
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()

#Train model for 10 iteations
train_model(10)



#Lets train with Stochastic Gradient Descent
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

LOSS_SGD = []
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

#Defining model with Stochastic Gradient Descent
def train_model_SGD(iter):
    
    #Epoch Loop
    for epoch in range(iter):
        
        #SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)

        #Store the loss 
        LOSS_SGD.append(criterion(Yhat, Y).tolist())
        
        for x, y in zip(X, Y):
            
            #make a pridiction
            yhat = forward(x)
        
            #calculate the loss 
            loss = criterion(yhat, y)

            #Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        
            #backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
        
            #update parameters slope and bias
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data

            #zero the gradients before running the backward pass
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()

#Run for 10 epochs
train_model_SGD(10)


#Lets Compare the two models
#Plot out the LOSS_BGD and LOSS_SGD

plt.plot(LOSS_BGD,label = "Batch Gradient Descent")
plt.plot(LOSS_SGD,label = "Stochastic Gradient Descent")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()


"""
SGD with Dataset DataLoader
"""

from torch.utils.data import Dataset, DataLoader


#Dataset Creation Class
class Data(Dataset):
    #Constructor
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = 1 * self.x - 1
        self.len = self.x.shape[0]
        
    #Getter
    def __getitem__(self,index):    
        return self.x[index], self.y[index]
    
    #Return the length
    def __len__(self):
        return self.len

#Create a Dataset object
dataset = Data()
print("The length of dataset: ", len(dataset)) #Using the len method in the Data class

#We can obtain value in the dataset with index numbers using the Getitem method in the class
x, y = dataset[0]
print(x," , ", y)

#Slicing the first 3 points
x, y = dataset[0:3]
print("The first 3 x: ", x)
print("The first 3 y: ", y)

get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

#Lets create the DataLoader object
trainloader = DataLoader(dataset = dataset, batch_size = 1)

w = torch.tensor(-15.0,requires_grad=True)
b = torch.tensor(-10.0,requires_grad=True)
LOSS_Loader = []

#Lets define our new function
#train_model_DataLoader
def train_model_DataLoader(epochs):
    
    #Epoch Loop
    for epoch in range(epochs):
        
        #SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        
        #store the loss 
        LOSS_Loader.append(criterion(Yhat, Y).tolist())
        
        for x, y in trainloader:
            
            #make a prediction
            yhat = forward(x)
            
            #calculate the loss
            loss = criterion(yhat, y)
            
            #Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            
            #Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            #Update parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr* b.grad.data
            
            #Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()

#Run for 10 epochs
train_model_DataLoader(10)


# Plot the LOSS_BGD and LOSS_Loader
plt.plot(LOSS_BGD,label="Batch Gradient Descent")
plt.plot(LOSS_Loader,label="Stochastic Gradient Descent with DataLoader")
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()


#Lets try another one with SGD and DataLoader
LOSS1 = []
w = torch.tensor(-12.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

def train_model_DataLoader1(epochs):
    
    #Epoch Loop
    for epoch in range(epochs):
        
        #SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
        Yhat = forward(X)
        
        #store the loss 
        LOSS1.append(criterion(Yhat, Y).tolist())
        
        for x, y in trainloader:
            
            #make a prediction
            yhat = forward(x)
            
            #calculate the loss
            loss = criterion(yhat, y)
            
            #Section for plotting
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            
            #Backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            #Update parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr* b.grad.data
            
            #Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()
            
        #plot surface and data space after each epoch    
        get_surface.plot_ps()

#Run for 10 epochs
train_model_DataLoader1(10)

plt.plot(LOSS1,label = "Stochastic Gradient Descent w/ DataLoader")
plt.plot(LOSS_BGD, color = 'orange', linestyle = 'dashed', label = "Batch Gradient Descent")
plt.xlabel('iteration')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()
