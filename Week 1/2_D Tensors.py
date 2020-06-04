# -*- coding: utf-8 -*-
"""

2-D Tensors

"""

"""
A 2d tensor can be viewed as a container the holds numerical values of the same type. 
In 2D tensors are essentially a matrix. Each row is a different sample and each column is a feature or attribute. 

We can also represent gray-scale images as 2D tensors. 
The image intensity values can be represented as numbers between 0 and 255. 0 corresponds to color black and 255 white.

Tensors can be extended to any number of dimensions. A 3D tensor is a combination of 3 1D tensors. 



"""


import torch
import numpy
import pandas

#Lets create a 2-D tensor
#We first create a list with 3 nested lists.
a = [[11,12,13], [21,22,23], [31,32,33]]
b = torch.tensor([[11,12,13], [21,22,23], [31,32,33]])

#We then cast the list to a torch tensor
A = torch.tensor(a)
print(A)

#Lets check the no. of dimensions or rank
print("Numer of dimensions of A = ", A.ndimension())

#The first list [] represents the first dimensions and the second represents the second dimension
#2D Tensors is as follows: [[]]

#Lets check the number of rows and columns of A. It should be 3,3 --- 3 rows, 3 columns
print("Shape of tensor A: ", A.shape)
#OR
print("Shape of tensor A: ", A.size())

#The 3,3 tensor has 2 axes. Axis = 0 (vertical) and Axis = 1 (Horizontal)

#Number of elements in a tensor -- using numel() method
print("Number of elements in A: ", A.numel())


#Indexing and Slicing 2D Tensors

#Indexing
print(A)
A[0][1] #Element in 1st row and 2nd column
A[1][2] #Element in 2nd row and 3rd column
A[2][0] #Element in 3rd row and first column

#Slicing
A[1:3,2] #Slicing elements in rows 2 and 3 from the 3rd column
A[2,0:3] #Slicing all the elements in the 3rd row


#Adding 2D tensors only works for tensors of the same type 
#Lets add A and B. Elements of the same position will be added
B = torch.tensor([[11,12,13], [21,22,23], [31,32,33]])
C = A + B
C

#Multiplication by a scalar is the same as multiplying a matrix by a scalr
#Multiplication of tensors is an elemenet-wise multiplication. Same position elements
D = A*B
print(D)


#Matrix multiplication can be done in torch but same rules will apply 
#First matrix must have equal columns to the rows of the second matrix

A = torch.tensor([[0,1,1],[1,0,1]])
B = torch.tensor([[1,1],[1,1],[-1,1]])

#Matrix multiplication is done by using the mm method
C = torch.mm(A,B)
print(C)
