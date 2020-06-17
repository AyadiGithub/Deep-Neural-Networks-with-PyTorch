"""

Hidden Layer Deep Network: Sigmoid, Tanh and Relu Activations Functions MNIST Dataset

"""

# In[1] Imports

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
torch.manual_seed(0)

# In[2] Neural Network Classes for Sigmoid, Tanh and Relu
# Neural Networks with two hidden layers


class NetworkSigmoid(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x


class NetworkTanh(nn.Module):
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x


class NetworkRelu(nn.Module):
    # Constructor
    def __init__(self, D_in, H1, H2, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# In[3] Training function


def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())

        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()

        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)

    return useful_stuff


# In[4] Using MNIST dataset, creating Training and Validation sets

# Creating the training_dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Creating the validation_dataset
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# In[6] Creating loss function

# criteiron loss function from torch.nn
criterion = nn.CrossEntropyLoss()

# In[7] Creating training and validation DataLoader objects

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# In[8] Setting up optimizer, model, learning_rate

input_dim = 28 * 28
hidden_dim1 = 50
hidden_dim2 = 50
output_dim = 10

# 28*28 pixels as input, 50 neurons in 1st layer and 50 in second, with 10 classes output
model = NetworkSigmoid(28*28, 50, 50, 10)

# 28*28 pixels as input, 50 neurons in 1st layer and 50 in second, with 10 classes output
model1 = NetworkTanh(28*28, 50, 50, 10)

# 28*28 pixels as input, 50 neurons in 1st layer and 50 in second, with 10 classes output
model2 = NetworkRelu(28*28, 50, 50, 10)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

# In[9] Training

# train sigmoid model
training_sigmoid = train(model, criterion, train_loader, validation_loader, optimizer, epochs=10)

# train tanh model
training_tanh = train(model1, criterion, train_loader, validation_loader, optimizer1, epochs=10)

# train relu model
training_relu = train(model2, criterion, train_loader, validation_loader, optimizer2, epochs=10)


# In[10] Plotting results

# Plotting training loss for different models
plt.plot(training_sigmoid['training_loss'], label='sigmoid')
plt.plot(training_tanh['training_loss'], label='tanh')
plt.plot(training_relu['training_loss'], label='relu')
plt.ylabel('loss')
plt.title('training loss iterations')
plt.legend()

# Plotting validation accuracy for different models
plt.plot(training_sigmoid['validation_accuracy'], label='sigmoid')
plt.plot(training_tanh['validation_accuracy'], label='tanh')
plt.plot(training_relu['validation_accuracy'], label='relu')
plt.ylabel('validation accuracy')
plt.xlabel('Iteration')
plt.legend()
