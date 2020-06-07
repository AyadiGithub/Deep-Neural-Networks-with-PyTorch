"""

Simple Linear Regression 


"""

'''
Manual Method

'''
import torch
import torch.nn as nn

w = torch.tensor(2.0, requires_grad = True) #Slope tensor. req_grad = true because they need to be learnt
b = torch.tensor(-1.0, requires_grad = True) #Bias tensor. req_grad = true because they need to be learnt
def forward(x): #Function that will predict y with a given x
    y = w * x + b
    return y

x = torch.tensor([1.0])    
yhat = forward(x) #yhat is an estimate of the real value of y
print(yhat)

x = torch.tensor([[1],[2]]) #2D tensor x
yhat = forward(x)
print(yhat)

'''
Using torch.nn Linear
'''

#Lets try another method
from torch.nn import Linear

torch.manual_seed(1) #The seed for generating random numbers = 1
model = Linear(in_features = 1, out_features = 1)
x = torch.tensor([0.0])
yhat = model(x.float())
print(yhat)

#Lets see the model parameters
print(list(model.parameters()))

x = torch.tensor([[1.0],[2.0]])
yhat = model(x)
print(yhat)
print(list(model.parameters()))



'''
Custom class
'''

#Lets create a sublass Linear Regression LR within nn.Module
#We user super to call a method from nn.Module without needing to intitalize it
class LR(nn.Module): #LR is defined as a child class of nn.Module
    def __init__(self, in_size, out_size): #Constructor
        #Inheriting methods from parent class nn.Module
        super(LR,self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    
    def forward(self, x): #Prediction funciton
        out = self.linear(x)
        return out

model = LR(1,1)

#We use state_dict to initialize the weight and bias of the model. 
#State_dict is a python dictionary. It can be used to map the relationship of linear layers to their parameters. 
model.state_dict()['linear.weight'].data[0] = torch.tensor([0.5153]) #Initializing weight
model.state_dict()['linear.bias'].data[0] = torch.tensor([-0.4414]) #initializing bias
print("The Parameters: ", list(model.parameters())) #Lets see the initialized weight and bias
#The above function works because the class LR, is a subclass of Modules. 

#Lets use the custom model to make predictions
x = torch.tensor([[1.0], [2.0]]) #2D tensor
print("The shape of x: ", x.shape)
yhat = model(x)
print(yhat)

x = torch.tensor([[1.0], [2.0], [3.0]])
print("The shape of x: ", x.shape)
print("The dimension of x: ", x.ndimension())
yhat = model(x)
print(yhat)


print("Python dictionary:", model.state_dict())
print("Keys:", model.state_dict().keys())
print("Values:", model.state_dict().values())





