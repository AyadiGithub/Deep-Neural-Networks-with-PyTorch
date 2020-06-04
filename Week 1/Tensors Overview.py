"""
1-D Tensors in Pytorch
------------------------------------------------------------------------------

Tenors are arrays that are the building blocks of a neural network. 

- A 0-D tenors is just a number: 1, 2, 0.2 , 10

1-D tensor is an array of numbers. It can be a row in a database, a vector, time series etc...
A tensor contains elements of a single data type. The Tensor Type is the type of tensor.
For real numbers: Tensor type is either a float or a double tensor or even half tensor.

32-bit float -                          Float Tensor
64-bit float -                          Double Tensor
16-bit float -                          Half Tensor
8-bit int (unsigned)(8-bit images)-     Byte Tensor     
8-bit int (signed)-                     Char Tensor
16-bit int (singed)-                    Short Tensor
32-bit int (singed)-                    Int Tensor
64-bit int (signed)-                    Long Tensor


"""


#To create a 1-D Tensor:

import torch

torch.__version__


a = torch.tensor([7,4,3,2,6]) # create a list and cast to pytorch tensor

#To find the datatype in the tensor:
print(a.dtype)    

#To find the type of tensor:
print(a.type())


#Lets create a float tensor
b = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

#To find the datatype in the tensor:
print(b.dtype)    

#To find the type of tensor:
print(b.type())


#Specifying data type of tensor within constructor also works
c = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

print(c.dtype)


#Explicitely creating a tensor of a specific type
d = torch.FloatTensor([0, 1, 2, 3, 4])
print(d.dtype)

#Converting tensor type
a = a.type(torch.FloatTensor)
print("New type of a: ", a.dtype)


#Checking tensor size. How many elements in the tensor?
print(a.size())

#Checking no. of dimensions of the tensor
print(c.ndimension())


#Converting a 1-D tensor to 2-D tensor by casting
a_col = a.view(5,1) #5 is no. of rows. 1 is number of columns
#If we dont know the number of elements (5) then we can use this argument
a_col = a.view(-1,1)

#Lets make sure it worked
print(a_col.ndimension())


#We can convert a numpy array to torch tensor this way
import numpy as np
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
torch_tensor = torch.from_numpy(numpy_array)
torch_tensor

#Converting torch tensor back to numpy array
back_to_numpy = torch_tensor.numpy()
back_to_numpy

#Converting Pandas series to tensor
import pandas as pd
pandas_series = pd.Series([0.1,2,0.3,10.1])
pandas_to_torch = torch.from_numpy(pandas_series.values)
pandas_to_torch

#Converting tensor to list
this_tensor = torch.tensor([0,1,2,3])
torch_to_list=this_tensor.tolist()
torch_to_list

'''

Individual values of tensors are also tensors

'''

new_tensor = torch.tensor([5,2,6,1])
new_tensor[0]
new_tensor[1]

#We can use .item to return the number in the tensor
new_tensor[0].item()
new_tensor[1].item()



'''

Indexing and slicing methods for Tensors

'''

c = torch.tensor([20,1,2,3,4])
#lets change the first tensor in C tensor to 0
c[0] = 0

print(c)

#slice - select elements in c and assign them to d
d = c[1:4]
print(d)

#Add or replace values in tensor c
c[3:5] = torch.tensor([5,6])


#Vector addition in pytorch
#Addition must be done with vectors of the same type
u = torch.tensor([1.0, 5.0])
v = torch.tensor([0.0, 3.0])
z = u + v
z = z.type(torch.FloatTensor)
print(z)


#Vector multiplicaiton with a Scalar
y = torch.tensor([1,2])
z = 2*y
print(z)

#Product of two tensors (element wise multiplication)
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
z = u*v
print(z)

#Dot product - element-wise multiplication and summation
y = torch.dot(u,v)
print(y) #y is still a tensor

#Adding a scalar value to a tensor aka. BROADCASTING
u = torch.tensor([1,2,3,-1])
z = u+1
print(z)



'''
Functions on tensors
'''

a = torch.tensor([1,-1,1,-1], dtype=float)
print(a.dtype)
mean_a = a.mean()
mean_a

b = torch.tensor([1, -1, 3, 4, 100])
MaxB = b.max()
MaxB


#Create a torch tensor in radiance using pi
np.pi
x = torch.tensor([0, np.pi/2, np.pi])
x

#Apply sin(x) to tensor x
y = torch.sin(x)
y

#Using linspace to plot mathematical functions
M = torch.linspace(-2, 2, steps=5) #-2 is starting point, 2 is ending, 5 is number of elements to generate
M

x = torch.linspace(0, 2*np.pi, 100) #Generating 100 evenly spaced elements from 0 to 2pi
x

y = torch.sin(x)
y


import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy()) #We need to convert tensors to numpy arrays





