"""

Convolution Neural Network Simple Example

"""

# In[1] Imports

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
torch.manual_seed(0)

# In[2] Plotting

# Plot function


def plot_channels(W):
    #number of output channels 
    n_out = W.shape[0]
    #number of input channels 
    n_in = W.shape[1]
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(n_out, n_in)
    fig.subplots_adjust(hspace=0.1)
    out_index = 0
    in_index = 0
    #plot outputs as rows inputs as columns 
    for ax in axes.flat:
    
        if in_index>n_in-1:
            out_index=out_index+1
            in_index=0
              
        ax.imshow(W[out_index,in_index,:,:], vmin = w_min, vmax = w_max, cmap = 'seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index = in_index+1

    plt.show()
    
# Plot data sample


def show_data(dataset, sample):

    plt.imshow(dataset.x[sample, 0, :, :].numpy(), cmap = 'gray')
    plt.title('y= '+str(dataset.y[sample].item()))
    plt.show()
  
    
# In[3] Create Data/Images

class Data(Dataset):
    
    def __init__(self, N_images=100, offset=0, p=0.9, train=False):
        """
        p: portability that pixel is white
        N_images: number of images 
        offset: set a random vertical and horizontal offset images by a sample (should be less than 3)
        
        """
       
        if train==True:
            np.random.seed(0)  
        
        # Make images multiple of 3 
        N_images = 2 * (N_images//2)
        images = np.zeros((N_images, 1, 11, 11))
        start1 = 3
        start2 = 1
        self.y = torch.zeros(N_images).type(torch.long)

        for n in range(N_images):
            if offset>0:
        
                low = int(np.random.randint(low = start1, high = start1+offset, size=1))
                high = int(np.random.randint(low = start2, high = start2+offset, size=1))
            else:
                low = 4
                high = 1
        
            if n <= N_images//2:
                self.y[n] = 0
                images[n, 0, high:high+9, low:low+3] = np.random.binomial(1, p, (9,3))
            elif  n > N_images//2:
                self.y[n] = 1
                images[n, 0, low:low+3, high:high+9] = np.random.binomial(1, p, (3,9))
           
        
        
        self.x = torch.from_numpy(images).type(torch.FloatTensor)
        self.len = self.x.shape[0]
        del(images)
        np.random.seed(0)
    
    
    def __getitem__(self, index):      
        return self.x[index], self.y[index]
    
    
    def __len__(self):
        return self.len


# In[4] Plot activations of the Convolutional Layer

def plot_activations(A, number_rows = 1, name = ""):
    A = A[0,:,:,:].detach().numpy()
    n_activations = A.shape[0]
    
    
    print(n_activations)
    A_min = A.min().item()
    A_max = A.max().item()

    if n_activations == 1:

        # Plot the image.
        plt.imshow(A[0,:], vmin = A_min, vmax = A_max, cmap = 'seismic')

    else:
        fig, axes = plt.subplots(number_rows, n_activations//number_rows)
        fig.subplots_adjust(hspace = 0.4)
        for i,ax in enumerate(axes.flat):
            if i< n_activations:
                # Set the label for the sub-plot.
                ax.set_xlabel( "activation:{0}".format(i+1))

                # Plot the image.
                ax.imshow(A[i,:], vmin = A_min, vmax = A_max, cmap = 'seismic')
                ax.set_xticks([])
                ax.set_yticks([])
    plt.show()


def conv_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    
    #by Duane Nielsen
    from math import floor
    
    if type(kernel_size) is not tuple:
        
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * padding) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * padding) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    
    return h, w


# In[4] Create Datasets

N_images = 10000
train_dataset = Data(N_images=N_images)

validation_dataset = Data(N_images=1000,train=False)
validation_dataset

show_data(train_dataset, 0)   

show_data(train_dataset, N_images//2+2)

out = conv_output_shape((11, 11), kernel_size=2, stride=1, padding=0, dilation=1)
print(out)

out1 = conv_output_shape(out, kernel_size=2, stride=1, padding=0, dilation=1)
print(out1)

out2 = conv_output_shape(out1, kernel_size=2, stride=1, padding=0, dilation=1)
print(out2)

out3 = conv_output_shape(out2, kernel_size=2, stride=1, padding=0, dilation=1)
print(out3)



# In[5] Create Conv2D Class


class CNN(nn.Module):
    def __init__(self,out_1=2,out_2=1):
        super().__init__()
        # First conv layer and maxpool
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=2, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Second conv layer and maxpool
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=2, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # Fully connected layer/ Linear layer
        self.fc1 = nn.Linear(out_2*7*7, 2)
     
        
    def forward(self,x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1) #Flattening maxpool2 output to a column
        x = self.fc1(x)
        return x
    
    
    def activations(self,x):
        # These are just for visualization
        z1 = self.cnn1(x)
        a1 = torch.relu(z1)
        out = self.maxpool1(a1)
        
        z2 = self.cnn2(out)
        a2 = torch.relu(z2)
        out = self.maxpool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out


# In[6] Create model, criterion loss function, optimizer, val and train loaders

model = CNN(2, 1)
model

# Plot the weights
plot_channels(model.state_dict()['cnn1.weight'])
plot_channels(model.state_dict()['cnn2.weight'])

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train and Val loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=10)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=20)


# In[7] Training loop

n_epochs = 10
cost_list = []
accuracy_list = []
N_test = len(validation_dataset)
cost = 0

#n_epochs
for epoch in range(n_epochs):
   
    cost=0    
    for x, y in train_loader:
        
        # Clear gradient 
        optimizer.zero_grad()
        
        # Prediction 
        z = model(x)
        
        # Calculate loss 
        loss = criterion(z,y)
        
        # Calculate gradients of parameters 
        loss.backward()
        
        # Update parameters 
        optimizer.step()
        
        cost += loss.item()
        
    cost_list.append(cost) # Add cost to cost_list
        
        
    correct = 0
    
    # Prediction on the validation data  
    
    for x_test, y_test in validation_loader:
        
        # Prediction
        z = model(x_test)
        
        # Take max value as prediction
        _,yhat = torch.max(z.data,1)

        # Sum the number of correct predictions (yhat==y_test)
        correct += (yhat==y_test).sum().item()
        

    accuracy = correct/N_test

    accuracy_list.append(accuracy)
    
    
# In[8] Analyze results and plot data

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('total loss', color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()


# Lets check the weights for each conv layer
model.state_dict()['cnn1.weight']
plot_channels(model.state_dict()['cnn1.weight'])

model.state_dict()['cnn1.weight']
plot_channels(model.state_dict()['cnn2.weight'])

show_data(train_dataset,N_images//2+2)

out = model.activations(train_dataset[N_images//2+2][0].view(1, 1, 11, 11))
out = model.activations(train_dataset[0][0].view(1, 1, 11, 11))

plot_activations(out[0], number_rows=1, name = " feature map")
plt.show()

plot_activations(out[2], number_rows=1, name = "2nd feature map")
plt.show()

plot_activations(out[3], number_rows=1, name = "first feature map")
plt.show()

out1 = out[4][0].detach().numpy()

out0 = model.activations(train_dataset[100][0].view(1,1,11,11))[4][0].detach().numpy()
out0

plt.subplot(2, 1, 1)
plt.plot( out1, 'b')
plt.title('Flatted Activation Values')
plt.ylabel('Activation')
plt.xlabel('index')
plt.subplot(2, 1, 2)
plt.plot(out0, 'r')
plt.xlabel('index')
plt.ylabel('Activation')
