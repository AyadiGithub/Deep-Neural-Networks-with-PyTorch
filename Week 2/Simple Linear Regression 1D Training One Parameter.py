"""

Linear Regression 1D: Training One Parameter

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

#Lets use this class for plotting and visualizing the parameter training
class plot_diagram():
    
    # Constructor
    def __init__(self, X, Y, w, stop, go = False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values] 
        w.data = start
        
    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y,'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)   
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
    
    # Destructor
    def __del__(self):
        plt.close('all')


#Lets generate some values that create a line with a slope of -3
X = torch.arange(-3, 3, 0.1).view(-1, 1) #View changes the shape of the tensor
f = -3 * X #Function of the line

#Lets plot the line
plt.plot(X.numpy(), f.numpy(), label = 'f') #We need to convert tensor to numpy
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Lets add noise to the data to simulate 'real data'. We use torch.randn to generate Gaussian noise.
Y = f + 0.1 * torch.randn(X.size()) #The noise must be the same size as X

#Lets plot Y
plt.plot(X.numpy(), Y.numpy(), label = 'Y') #We need to convert tensor to numpy
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



#Create the model and cost function(total loss)
#Create a forward function for prediction
def forward(x):
    return w * x

#Mean Squared Error function for evaluating the result
def criterion(yhat, y):
    return torch.mean((yhat - y)**2)

#Create a learning rate
lr = 0.1

#Create an empty list to append loss results for each iteration
LOSS = []

#We create a parameter w with requires_grad = True to indicate that torch must learn it
w = torch.tensor(-10.0, requires_grad = True) #Initialize w as -10.0

#Create a plot diagram object to visualize the data and the parameter for each iteration
gradient_plot = plot_diagram(X, Y, w, stop = 5)


#Creating a function to train the model
def train_model(iter):
    for epoch in range (iter):
        
        #Prediction Yhat using forward function
        Yhat = forward(X)
        
        #loss calculation using criterion loss function on Yhat, Y
        loss = criterion(Yhat,Y)
        
        #Plotting diagram for visualization
        gradient_plot(Yhat, w, loss.item(), epoch)
        
        #Appending Loss to LOSS list
        LOSS.append(loss.item())
        
        #Compute the gradient of the loss wrt all parameters
        loss.backward()
        
        #Update parameters
        w.data = w.data - lr * w.grad.data
        
        #Zero the gradients before running the backward pass
        w.grad.data.zero_()
        
#Lets train the model for 4 iterations
train_model(4)        

#Plotting the list LOSS (loss per iteration)        
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")        
    
    
#Lets try a new parameter w
w = torch.tensor(-15.0, requires_grad=True) #Initialize w as -15.0

#Create an empty list to store the loss
LOSS2 = []
gradient_plot1 = plot_diagram(X, Y, w, stop = 15)


def my_train_model(iter):
    for epoch in range (iter):
        
        #Prediction Yhat using forward function
        Yhat = forward(X)
        
        #loss calculation using criterion loss function on Yhat, Y
        loss = criterion(Yhat,Y)
        
        #Plotting diagram for visualization
        gradient_plot1(Yhat, w, loss.item(), epoch)
        
        #Appending Loss to LOSS list
        LOSS2.append(loss.item())
        
        #Compute the gradient of the loss wrt all parameters
        loss.backward()
        
        #Update parameters
        w.data = w.data - lr * w.grad.data
        
        #Zero the gradients before running the backward pass
        w.grad.data.zero_()


my_train_model(4)

plt.plot(LOSS, label = "LOSS")
plt.plot(LOSS2, label = "LOSS2")
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.legend()


#We notice that the parameter value is sensitive to initialization.