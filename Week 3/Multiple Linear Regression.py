"""

Multiple Linear Regression


"""
from torch import nn
import torch
torch.manual_seed(1)

#Setting weight and bias manually
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)
print(w)
print(b)

#Define Prediction Function
def forward(x):
    yhat = torch.mm(x, w) + b #torch.mm is matrix multiplicaiton. torch.matmul can also be used here. 
    return yhat

x = torch.tensor([[1.0, 2.0]])
print("Dimensions of x: ", x.ndimension())

yhat = forward(x)
print("The result: ", yhat)

#Sample tensor X. Lets create a 2D X with 3 samples (or rows)
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

#Make the prediction of X 
yhat = forward(X)
print("The result: ", yhat)

#Make a linear regression model using build-in function
model = nn.Linear(2, 1)

#Make a prediction of x
yhat = model(x)
print("The result: ", yhat)

#Model auto initialized parameters
print("model initialized parameters: ", model.state_dict())

#Make a prediction of X
yhat = model(X)
print("The result: ", yhat)


#Create linear_regression Class using nn.Module Class
class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat
 
#Create multiple linear regression model with 2 inputs and 1 output
model = linear_regression(2, 1)

#Print model parameters (auto initialized)
print("The parameters: ", list(model.parameters()))

#Print model parameters
print("The parameters: ", model.state_dict())


#Make a prediction of x
yhat = model(x)
print("The result: ", yhat)


#Make a prediction of X
yhat = model(X)
print("The result: ", yhat)



#Lets try a different X
X = torch.tensor([[11.0, 12.0, 13, 14], [11, 12, 13, 14]])

#Since the X created has x1, x2, x3, x4 and 2 samples. We need to redo our model to be 4inputs and 1 ouput.
model = linear_regression(4,1)

#Prediction
yhat = model(X)
print(yhat)






