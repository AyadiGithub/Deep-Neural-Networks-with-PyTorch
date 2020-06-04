
"""

1-D Tensors
Exercise




32-bit float -                          Float Tensor
64-bit float -                          Double Tensor
16-bit float -                          Half Tensor
8-bit int (unsigned)(8-bit images)-     Byte Tensor     
8-bit int (signed)-                     Char Tensor
16-bit int (singed)-                    Short Tensor
32-bit int (singed)-                    Int Tensor
64-bit int (signed)-                    Long Tensor


"""

import torch 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#Lets define the function to plot vectors in coorindate system
def plotVec(vectors):
    ax = plt.axes()

# Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]    
    #For loop for drawing vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width = 0.05, color = vec["color"], head_length = 0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])
    
    plt.ylim(-2, 2) #Setting limits for y-axis
    plt.xlim(-2, 2) #Setting limits for x-axis



# Convert a integer list with length 5 to a tensor

ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())
type(ints_to_tensor)


#Convert float list to a float tensor 32bit
list_floats = [0.0, 1.0, 2.0, 3.0, 4.0]
list_floats = torch.tensor(list_floats)
list_floats.type()

#convert the float tensor to int64 Long Tensor
floats_int_tensor=torch.tensor(list_floats,dtype=torch.int64)
floats_int_tensor.type()


# Convert a integer list with length 5 to float tensor
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print("The type of the new_float_tensor:", new_float_tensor.type())
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])

# Another method to convert the integer list to float tensor
old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
print("The type of the new_float_tensor:", new_float_tensor.type())


# Introduce the tensor_obj.size() & tensor_ndimension.size() methods
print("The size of the new_float_tensor: ", new_float_tensor.size())
print("The dimension of the new_float_tensor: ",new_float_tensor.ndimension())


'''
The tensor_obj.view(row, column) is used for reshaping a tensor object.

What if you have a tensor object with torch.Size([5]) as a new_float_tensor as shown in the previous example?
After you execute new_float_tensor.view(5, 1), the size of new_float_tensor will be torch.Size([5, 1]).
This means that the tensor object new_float_tensor has been reshaped from 
a one-dimensional tensor object with 5 elements to a two-dimensional tensor object with 5 rows and 1 column.

'''

# Introduce the tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)
print("Size after view method", twoD_float_tensor.size()) #Tensor of size (5,1)
print("Size after view method", twoD_float_tensor.ndimension()) #Tensor dimension is now 2

#What if you have a tensor with dynamic size but you want to reshape it? You can use -1 to do just that.
# Introduce the use of -1 in tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)

# Convert a numpy array to a tensor
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

# Convert a tensor to a numpy array
back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

# Set all elements in numpy array to zero 
numpy_array[:] = 0
print("The new tensor points to numpy_array : ", new_tensor)
print("and back to numpy array points to the tensor: ", back_to_numpy)


#Converting a tensor to (1,5) tensor
your_tensor = torch.tensor([1, 2, 3, 4, 5])
your_tensor.size()
your_tensor.ndimesnion()
your_new_tensor = your_tensor.view(1, 5)
your_new_tensor.ndimension()
print("Original Size: ", your_tensor)
print("Size after view method", your_new_tensor)



import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy()) #We need to convert tensors to numpy arrays


#Constructing tensor with 25 steps in the range of 0 and pi/2.
x = torch.linspace(0, np.pi/2, steps=25)
y = torch.sin(x)

plt.plot(x.numpy(), y.numpy())

print("Max value from tensor x = ", x.max())
print("Min value from tensor x = ", x.min())

#Convert the list [-1, 1] and [1, 1] to tensors u and v. 
#Plot the tensor u and v as a vector by using the function plotVec and find the dot product

u = torch.tensor([-1, 1])
v = torch.tensor([1, 1])
z = torch.dot(u,v)
print("Dot product of tensors u and v = ", np.dot(u,v))


plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'}
])







