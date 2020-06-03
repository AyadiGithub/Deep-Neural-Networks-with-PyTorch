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












