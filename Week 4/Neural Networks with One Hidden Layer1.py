"""

Neural Networks with One Hidden Layer

"""

# In[1]: Imports

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

# In[2]
#Plotting function
def plot_accuracy_loss(training_results):
    plt.subplot(2, 1, 1)
    plt.plot(training_results['training_loss'], 'r')
    plt.ylabel('loss')
    plt.title('training loss iterations')
    plt.subplot(2, 1, 2)
    plt.plot(training_results['validation_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')   
    plt.show()    

# In[3]
#Model Parameters function
def print_model_parameters(model):
    count = 0
    
    for element in model.state_dict():
        count += 1
        if count %2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if element.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[element].size())
        else:
            print("The size of weights: ", model.state_dict()[element].size())
       
# In[4]
#Function to show data (images)
def show_data(data_sample):
    plt.imshow(data_sample.numpy().reshape(28, 28), cmap = 'gray')
    plt.show()            
             
# In[5]
#Class Neural Net with 1 hidden layer
class SimpleNet(nn.Module):
    #Constructor
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H) #input layer
        self.linear2 = nn.Linear(H, D_out) #Output layer
        
    #Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x) #We want to predict multiple classes so sigmoid function shouldnt be used
        
        return x
        
# In[6]
#Training function
def train_model(model, criterion, train_loader, validation_loader, optimizer, epochs = 100):
    i = 0
    useful_stuff = {'training_loss': [], 'validation_accuracy': []}
    
    for epoch in range(epochs):
        
        for i, (x, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            z = model(x.view(-1, 28*28)) #reshaping to 28*28 
            
            loss = criterion(z, y)
            
            loss.backward()
        
            optimizer.step()
            
            useful_stuff['training_loss'].append(loss.data.item())
     
        correct = 0
        
        for x, y in validation_loader:
            
            z = model(x.view(-1, 28*28)) #reshaping to 28*28
            
            _, label = torch.max(z, 1) #Take class with max probability
            
            correct += (label == y).sum().item() 
            
        accuracy = 100 * (correct / len(validation_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff
    
    
# In[7]

#Load Data 
train_dataset = dsets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor()) #Load train MNIST set and transform to tensor
validation_dataset = dsets.MNIST(root='./data', train = False, download = True, transform = transforms.ToTensor()) #Load non-train MNIST set and transform to tensor

# In[8]
#Create dataloader, criterion function, optimizer, learning rate

#traind and validation loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

#Criterion function
criterion = nn.CrossEntropyLoss()

#learning rate
learning_rate = 0.01

#Create model with input dimension of the images WxH 28*28, 100 neurons and 10 output dim.
model = SimpleNet(784, 100, 10)

#Model parameters
print_model_parameters(model)

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# In[9]

#Training the model

training_results = train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=30)


# In[10]
#Plot accuracy and loss

plot_accuracy_loss(training_results)


#Plot the 1st 5 misclassified items
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _,yhat = torch.max(z, 1)
    if yhat != y:
        show_data(x)
        count += 1
    if count >= 5:
        break



# In[11]

#Use nn.Sequential to build to same model, train it and plot
model = torch.nn.Sequential(nn.Linear(784, 100), nn.Sigmoid(), nn.Linear(100, 10))

training_results = train_model(model, criterion, train_loader, validation_loader, optimizer, epochs = 10)

plot_accuracy_loss(training_results)






















