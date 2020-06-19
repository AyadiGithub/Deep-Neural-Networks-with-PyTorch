"""

Convolutional Neural Network with Small Images

"""

# In[1] Imports

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# In[2] Plot function

# Define the function for plotting the channels

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = ' + str(data_sample[1].item()))


# In[3] Create Data


IMAGE_SIZE = 16

# Create a transform to resize image and convert to tensor
transformation = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# Create Dataset from MNIST and apply composed transformation
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transformation)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transformation)

# Check Data
train_dataset[0][1]  # Label - int object
train_dataset[0][0]
train_dataset[0][0].size()
train_dataset[0][0].type()

train_dataset[1][1]
train_dataset[1][0]
train_dataset[1][0].size()
train_dataset[1][0].type()

# Lets plot some images.
# We need to squeeze the (1, 16, 16) to (16, 16) and convert to numpy array
# We can also use reshape(16, 16)
plt.imshow(train_dataset[1][0].squeeze().numpy())
plt.imshow(train_dataset[5][0].reshape(16, 16).numpy())
# Plot in gray scale
plt.imshow(train_dataset[5][0].reshape(16, 16).numpy(), cmap='gray')

# We can use show_data function
show_data(train_dataset[1][0])


# In[4] Create CNN Class


class CNN(nn.Module):

    # Constructor
    def __init__(self, out_1=16, out_2=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.ELU = nn.ELU()
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2*4*4, 10)

    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.ELU(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    # Outputs in each step
    def activations(self, x):
        # This part is for visualization purposes
        z1 = self.cnn1(x)
        a1 = self.ELU(z1)
        out = self.maxpool1(a1)
        z2 = self.cnn2(out)
        a2 = self.ELU(z2)
        out1 = self.maxpool2(a2)
        out2 = out1.view(out1.size(0), -1)
        return z1, a1, out, z2, a2, out1, out2


# In[5] Train loop and training


def train_model(model, train_loader, validation_loader, optimizer, n_epochs=4):

    # Global variable
    N_test = len(validation_dataset)
    accuracy_list = []
    loss_list = []
    for epoch in range(n_epochs):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)

        correct = 0
        # Perform a prediction on the validation data
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)

    return accuracy_list, loss_list



# In[6] Batch Normalization CNN class


class CNN_BatchNorm(nn.Module):
    # Constructor
    def __init__(self, out_1=16, out_2=32):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)  # To normalize conv2D, we need BatchNorm2D
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.ELU = nn.ELU()
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2*4*4, 10)
        self.bn_fc1 = nn.BatchNorm1d(10)  # To normalize linear layer, BatchNorm1D must be used

    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.ELU(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = self.ELU(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x


# In[7] Initialize, create loss function, optimizer and data loaders


# Create model object from CNN class
model = CNN(16, 32)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Loss function criterion
criterion = nn.CrossEntropyLoss()

# train and val loader
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=5000)

accuracy_list_normal, loss_list_normal = train_model(model=model,
                                                     n_epochs=10, train_loader=train_loader, validation_loader=validation_loader,
                                                     optimizer=optimizer)


# Create model_BatchNorm object from CNN_BatchNorm class
model_BatchNorm = CNN_BatchNorm(16, 32)

# Optimizer
optimizer_BatchNorm = torch.optim.Adam(model_BatchNorm.parameters(), lr=0.01)

accuracy_list_batch, loss_list_batch = train_model(model=model_BatchNorm,
                                                   n_epochs=10, train_loader=train_loader, validation_loader=validation_loader,
                                                   optimizer=optimizer_BatchNorm)


# In[8] Analyze Results and Compare

# Plot the loss and accuracy
plt.plot(loss_list_normal, 'b', label='loss normal cnn')
plt.plot(loss_list_batch, 'r', label='loss batch cnn')
plt.xlabel('iteration')
plt.title("loss")
plt.legend()

plt.plot(accuracy_list_normal, 'b', label='normal CNN')
plt.plot(accuracy_list_batch, 'r', label='CNN with Batch Norm')
plt.xlabel('Epoch')
plt.title("Accuracy ")
plt.legend()
plt.show()
