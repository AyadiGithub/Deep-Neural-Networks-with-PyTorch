"""

Mini-Batch Gradient Descent with DataLoader

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
get_surface = plot_error_surfaces(15, 13, X, Y, 30, go = False)

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

"""

Mini-Batch with batch size = 5

"""

#Create a Dataset object
dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5) #Selecting a batch size of 5. 
print("The length of dataset: ", len(dataset)) #Using the len method in the Data class

# Define the parameters w, b for y = wx + b
w = torch.tensor(-15.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.
b = torch.tensor(-10.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.

#Define a learning rate and create an empty list LOSS to store the loss for each iteration
lr = 0.1 #Our first choice learning rate
LOSS_MINI5 = []  #Create the MiniBatch Loss empty list

def train_model_Mini5(epochs):
    
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()
        LOSS_MINI5.append(criterion(forward(X), Y).tolist())
       
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_Mini5(5)


"""

Mini-Batch with batch size = 10

"""

#Create a Dataset object
dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 10) #This time we change batch size to 10
print("The length of dataset: ", len(dataset)) #Using the len method in the Data class

# Define the parameters w, b for y = wx + b
w = torch.tensor(-15.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.
b = torch.tensor(-10.0, requires_grad = True) #Requires_grad = true because torch needs to learn it.

#Define a learning rate and create an empty list LOSS to store the loss for each iteration
lr = 0.1 #Our first choice learning rate
LOSS_MINI10 = []  #Create the MiniBatch Loss empty list

def train_model_Mini10(epochs):
    
    for epoch in range(epochs):
        Yhat = forward(X)
        get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), criterion(Yhat, Y).tolist())
        get_surface.plot_ps()
        LOSS_MINI10.append(criterion(forward(X), Y).tolist())
       
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            get_surface.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

train_model_Mini10(5)



#Lets Compare the two batch sizes
#Plot out the LOSS_MINI10 and LOSS_MINI5

plt.plot(LOSS_MINI10,label = " Mini Batch Gradient Descent 10")
plt.plot(LOSS_MINI5, label = " Mini Batch Gradient Descent 5", linestyle='dashed')
plt.xlabel('epoch')
plt.ylabel('Cost/ total loss')
plt.legend()
plt.show()




