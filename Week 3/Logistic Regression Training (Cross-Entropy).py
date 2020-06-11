"""

Logistic Regression Training Negative Log likelihood (Cross-Entropy)

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

#Create class and function for plotting
class plot_error_surfaces(object):
    
    #Construstor
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
                Z[count1, count2] = np.mean((self.y - (1 / (1 + np.exp(-1*w2 * self.x - b2)))) ** 2)
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
            plt.figure(figsize=(7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
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
            
     #Setter
    def set_para_loss(self, model, loss):
        self.n = self.n + 1
        self.W.append(list(model.parameters())[0].item())
        self.B.append(list(model.parameters())[1].item())
        self.LOSS.append(loss)
    
    #Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W, self.B, self.LOSS, c='r', marker='x', s=200, alpha=1)
        plt.figure()
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
        
    #Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label="estimated line")
        plt.plot(self.x, 1 / (1 + np.exp(-1 * (self.W[-1] * self.x + self.B[-1]))), label='sigmoid')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-0.1, 2))
        plt.title('Data Space Iteration: ' + str(self.n))
        plt.show()
        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c='r', marker='x')
        plt.title('Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        
#Plot the diagram
def PlotStuff(X, Y, model, epoch, leg=True):
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    if leg == True:
        plt.legend()
    else:
        pass
    
#Create a Data class
class Data(Dataset):
    #Constructor
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1) #Create a column of values between -1 and 1 with 0.1 step
        self.y = torch.zeros(self.x.shape[0], 1) #Create a column of zeros with length of x
        self.y[self.x[:, 0] > 0.2 ] = 1 #y = 1 when x > 0.2
        self.len = self.x.shape[0]
        
    #Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    #Length
    def __len__(self):
        return self.len

#Create the dataset
data_set = Data()
print(data_set.x) #Lets see x
print(data_set.y) #Lets see y

#Create a logistic regression custom class using nn.Module
class logistic_regression(nn.Module):
    
    #Constructor
    def __init__( self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    #Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        
        return yhat


#Initialize model
model = logistic_regression(1) #1 input model

#Manually setting weight and bias
model.state_dict() ['linear.weight'].data[0] = torch.tensor([[-5]])
model.state_dict() ['linear.bias'].data[0] = torch.tensor([[-10]])
print("The parameters: ", model.state_dict())


#Create the plot_error_surfaces object
get_surface = plot_error_surfaces(15, 13, data_set[:][0], data_set[:][1], 30)


#Create a DataLoader object
trainloader = DataLoader(dataset = data_set, batch_size = 3)

#Create the error metric. Using RMS will cause the manually initialized parameters not to converge
#Cross Entropy Loss error will be used to solve this problem
#We can either manually create a loss function or import from torch

#Lets create a cross entropy loss function
def criterion(yhat, y):
    out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat)) #negative log cross entropy function
    
    return out

#This is the built in cross entropy function in torch
#criterion = nn.BCELoss()

#Set the learning rate
learning_rate = 2

#Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#Create the training loop
def train_model(epochs):
    for epoch in range(epochs):
        
        for x, y in trainloader:
            yhat = model(x) #predict
            loss = criterion(yhat, y) #calculate loss
            optimizer.zero_grad() #Set gradients to zero
            loss.backward() #Calculate gradients
            optimizer.step() #adjust new parameters
            get_surface.set_para_loss(model, loss.tolist()) #plot
        
        if epoch % 20 == 0:
            get_surface.plot_ps()

train_model(100)

#Prediction
yhat = model(data_set.x)
label = yhat > 0.5
print("The accuracy: ", torch.mean((label == data_set.y.type(torch.ByteTensor)).type(torch.float))) #Accuracy = 1 or 100%
















