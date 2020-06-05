"""

Differentiation in Pytorch

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

#Evaluation of derivatives in torch
x = torch.tensor(2, requires_grad=True, dtype=float) #requires_grad tells torch that x value will be used to evaluate functions
y = x**2
#To calculate derivative of y
y.backward() #Backward function Calculates the derivative of y wrt. to x
print(x.grad) 


#Lets create a new tensor z and evaluate its derivative wrt x
x = torch.tensor(2, requires_grad=True, dtype=float)
z = x**2 + 2*x + 1
z.backward()
print(x.grad)


#Partial Derivatives
u = torch.tensor(1, requires_grad=True, dtype=float)
v = torch.tensor(2, requires_grad=True, dtype=float)
f = u*v + u**2
f.backward()
print(u.grad)
print(v.grad)


##############################################################################

# Create a tensor x
x = torch.tensor(2.0, requires_grad = True)
print("The tensor x: ", x)

# Create a tensor y according to y = x^2
y = x ** 2
print("The result of y = x^2: ", y)

# Take the derivative. Try to print out the derivative at the value x = 2
y.backward()
print("The dervative at x = 2: ", x.grad)

#Below are the attributes of x and y that torch creates. 
print('data:',x.data)
print('grad_fn:',x.grad_fn)
print('grad:',x.grad)
print("is_leaf:",x.is_leaf)
print("requires_grad:",x.requires_grad)

print('data:',y.data)
print('grad_fn:',y.grad_fn)
print('grad:',y.grad)
print("is_leaf:",y.is_leaf)
print("requires_grad:",y.requires_grad)


#We can implement our own custom autograd Functions by subclassing torch.autograd.Function 
#And implementing the forward and backward passes which operate on Tensors

class SQ(torch.autograd.Function):


    @staticmethod
    def forward(ctx,i):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        result=i**2
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        i, = ctx.saved_tensors
        grad_output = 2*i
        return grad_output


#Lets apply the function
x=torch.tensor(2.0,requires_grad=True )
sq=SQ.apply
y=sq(x)
y
print(y.grad_fn)
y.backward()
x.grad


# Calculate the derivative with multiple values
x = torch.linspace(-10, 10, 10, requires_grad = True)
Y = x ** 2
y = torch.sum(x ** 2)

# Take the derivative with respect to multiple value. Plot out the function and its derivative
y.backward()
#The method detach()excludes further tracking of operations in the graph, and therefore the subgraph will not record operations
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()

# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative
x = torch.linspace(-10, 10, 1000, requires_grad = True)
Y = torch.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
print(y.grad_fn)
