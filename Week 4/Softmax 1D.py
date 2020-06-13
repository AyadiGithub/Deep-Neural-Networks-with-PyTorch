"""

Softmax Classifier 1D

"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset

#Create a plotting function
def plot_data(data_set, model = None, n = 1, color = False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label = 'y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label = 'y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label = 'y = 2')
    plt.ylim((-0.1, 3))
    plt.legend()
    if model != None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['0.weight'], model.state_dict()['0.bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label = y_l)
        if color == True:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where = y1 > y1, interpolate = True, color = 'blue')
            plt.fill_between(x, y0, where = y1 > y2, interpolate = True, color = 'blue')
            plt.fill_between(x, y1, where = y1 > y0, interpolate = True, color = 'red')
            plt.fill_between(x, y1, where = ((y1 > y2) * (y1 > y0)),interpolate = True, color = 'red')
            plt.fill_between(x, y2, where = (y2 > y0) * (y0 > 0),interpolate = True, color = 'green')
            plt.fill_between(x, y2, where = (y2 > y1), interpolate = True, color = 'green')
    plt.legend()
    plt.show()


torch.manual_seed(1)

#Create Data class using torch Dataset
class Data(Dataset):
    
    #constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
        
    #Get function
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
    
    def __len__(self):
         return self.len

#Create Data object
data_set = Data()

#Plot data_set
plot_data(data_set)

"""
Build a Softmax Classifier
"""

model = nn.Sequential(nn.Linear(1, 3)) #Create a Sequential model with 1 input and 3 outputs
model.state_dict() #Show initialized parameters


#Create criterion loss function, optimizer, and Dataloader object
criterion = nn.CrossEntropyLoss()

#Optimizer Stochastic Gradient Descent with learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

#Dataloader object with batch size 5
trainloader = DataLoader(dataset = data_set, batch_size = 5)


"""
Training Model
"""

#Create empty Loss list
LOSS = []

#Create training function
def train_model(epochs):
    for epoch in range(epochs):
        
        if epoch %50 == 0: #When remainder of epoch/50 = 0, pass and plot 
            pass
            plot_data(data_set, model) #Plot every 50 epochs
        
        for x, y in trainloader:
            optimizer.zero_grad()
            
            yhat = model(x)
            
            loss = criterion(yhat, y)
           
            LOSS.append(loss)
           
            loss.backward()
           
            optimizer.step()

train_model(300) #train model for 300 epochs

"""
Analyze results and Predict
"""

#Model prediction z
z = model(data_set.x)
_, yhat = z.max(1)
print("The prediction", yhat)


#Accuracy
correct = (data_set.y == yhat).sum().item()
accuracy = correct / len(data_set)
print("The accuracy: ", accuracy)


#Softmax function can also convert 'z' to a probability
Softmax_f = nn.Softmax(dim = -1)
Probability = Softmax_f(z)

for i in range(3):
    print("probability of class {} is given by  {}".format(i, Probability[0,i]) )































