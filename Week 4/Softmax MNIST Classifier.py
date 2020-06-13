"""

Softmax Classifier using MNIST numbers dataset

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


#Define a plotting function to plot parameters
def PlotParameters(model): 
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            
            # Set the label for the sub-plot.
            ax.set_xlabel("class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()
    
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray') #Showing image 28x28 in gray scale
    plt.title('y = ' + str(data_sample[1].item())) #Showing title as label corresponding to image




#Lets Create some data (training set from mnist)
train_dataset = dsets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor())
print("The Training dataset:\n", train_dataset)

#Create validation set
val_dataset = dsets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())
print("The Validation dataset:\n", val_dataset)

#Print datatype
print("Type of data element: ", train_dataset[0][1].type())

#Plotting image samples
print("The image: ", show_data(train_dataset[3]))
print("The image: ", show_data(train_dataset[2]))



"""
Build a Softmax Classifier Class
"""
#Softmax Class from nn.Module
class SoftMax(nn.Module):
    
    #Constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    #Prediction
    def forward(self, x):
        
        z = self.linear(x)
        
        return z

#train_dataset shape
train_dataset[0][0].shape

#The trainset needs to be flattened to 1 column and multiple rows (728)

input_dim = 28 * 28 #728 rows 
output_dim = 10 #10 Categories 0,1,2,3,4,5,6,7,8,9


#Create the model
model = SoftMax(input_dim, output_dim)
print("The Model:\n", model)


#Lets see the initialized parameters and their size
print('W: ',list(model.parameters())[0].size())
print('b: ',list(model.parameters())[1].size())
print("The Parameters are: \n", model.state_dict())


#Lets plot the parameters
PlotParameters(model)

#Load data into DataLoader
train_loader = DataLoader(dataset = train_dataset, batch_size = 100)
val_loader = DataLoader(dataset = val_dataset, batch_size = 5000)

#Set the learning_rate
learning_rate = 0.1

#Define optimizer and criterion
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()



#Train the model
epochs = 10
LOSS_List = [] #empty list to store LOSS
Accuracy_List = []
N_test = len(val_dataset)

#Training function
def train_model(n_epochs):
    for epoch in range(epochs):
        
        for x, y in train_loader:
           
            optimizer.zero_grad()
           
            z = model(x.view(-1, 28 * 28)) #reshaping to 28x28
            
            loss = criterion(z, y)
            
            loss.backward()
           
            optimizer.step()
            
        correct = 0
        print(correct)
        
        #Perform a prediction on the validation data  
        for x_test, y_test in val_loader:
           
            z = model(x_test.view(-1, 28 * 28))
           
            _, yhat = torch.max(z.data, 1) #Take max value from z 
           
            correct += (yhat == y_test).sum().item()
            print(correct)
       
        accuracy = correct / N_test
       
        LOSS_List.append(loss.data)
      
        Accuracy_List.append(accuracy)

train_model(epochs)


"""
Analyze the model
"""

#Plot the loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(LOSS_List,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot(Accuracy_List, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()


#Plot trained parameters
PlotParameters(model)


#Plot missclassified examples
Softmax_function = nn.Softmax(dim = -1)
count = 0

#Plot the first 5 correctly classified samples and their respective probability by using torch.max 
for x,y in val_dataset:
    
    z = model(x.reshape(-1, 28*28))
    _, yhat = torch.max(z, 1)
    
    if yhat == y:
        show_data((x, y))
        plt.show()
        print("yhat: ". yhat)
        print("Probability of class: ", torch.max(Softmax_function(z).item()))
        count += 1
    
    if count >= 5:
        break


#Plot the first 5 missclassified samples and their respective probability by using torch.max 
count = 0
for x,y in val_dataset:
    
    z = model(x.reshape(-1, 28*28))
    _, yhat = torch.max(z, 1)
    
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat: ". yhat)
        print("Probability of class: ", torch.max(Softmax_function(z).item()))
        count += 1
    
    if count >= 5:
        break










