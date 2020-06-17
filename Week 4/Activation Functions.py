"""

Activation Functions

"""

# In[1]: Imports

# Import the libraries we need for this lab

import torch.nn as nn
import torch

import matplotlib.pyplot as plt
torch.manual_seed(0)

# In[2]:
# Create a tensor

z = torch.arange(-10, 10, 0.1,).view(-1, 1)

# Create a sigmoid object with torch.nn

sig = nn.Sigmoid()

# Make a prediction of sigmoid function

yhat = sig(z)

# Plot the result


def plot_sig():
    plt.plot(z.numpy(), yhat.detach().numpy())
    plt.xlabel('z')
    plt.ylabel('yhat')
    plt.show()


plot_sig()

# Use the built-in torch function to predict the result
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


# Create a tanh object with torch.nn
TANH = nn.Tanh()


# Make the prediction using tanh object


def plot_tanh():
    yhat = TANH(z)
    plt.plot(z.numpy(), yhat.numpy())
    plt.xlabel('z')
    plt.ylabel('yhat')
    plt.show()


plot_tanh()
# Make the prediction using the build-in tanh object
yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()


# Create a relu object and make the prediction

def ReLU():
    RELU = nn.ReLU()
    yhat = RELU(z)
    plt.plot(z.numpy(), yhat.numpy())
    plt.xlabel('z')
    plt.ylabel('yhat')
    plt.show()


ReLU()

# Use the build-in function to make the prediction
yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# Plot the results to compare the activation functions
x = torch.arange(-2, 2, 0.1).view(-1, 1)


def Plot_all():
    plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
    plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
    plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
    plt.legend()
    plt.show()


Plot_all()

# Compare the activation functions again using a tensor in the range (-1, 1)
x = torch.arange(-1, 1, 0.1).view(-1, 1)

Plot_all()






































