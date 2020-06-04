
"""

2-D Tensors
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

import numpy as np 
import matplotlib.pyplot as plt
import torch
import pandas as pd

#Creating 2D tensor from a list
twoD_list = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
twoD_tensor = torch.tensor(twoD_list)
print("The New 2D Tensor: ", twoD_tensor)

#Converting tensor to numpy array and back to tensor
twoD_numpy = twoD_tensor.numpy()
print("Tensor -> Numpy Array:")
print("The numpy array after converting: ", twoD_numpy)
print("Type after converting: ", twoD_numpy.dtype)

new_twoD_tensor = torch.from_numpy(twoD_numpy)
print("Numpy Array -> Tensor:")
print("The tensor after converting:", new_twoD_tensor)
print("Type after converting: ", new_twoD_tensor.dtype)

#Lets convert panadas dataframe to a tenosr
df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})

print("Pandas Dataframe to numpy: ", df.values)
print("Type BEFORE converting: ", df.values.dtype)


new_tensor = torch.from_numpy(df.values)
print("Tensor AFTER converting: ", new_tensor)
print("Type AFTER converting: ", new_tensor.dtype)

#Lets convert a Pandas Series to a tensor
df = pd.DataFrame({'A':[11, 33, 22],'B':[3, 3, 2]})
pandas_to_numpy = df.values
numpy_to_tensor = torch.tensor(pandas_to_numpy)
numpy_to_tensor



#Slicing rows
tensor_example = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
sliced_tensor_example = tensor_example[1:3] #Slicing 2nd and 3rd row
print("Result after tensor_example[1:3]: ", sliced_tensor_example)
print("Dimension after tensor_example[1:3]: ", sliced_tensor_example.ndimension())

#Dimension of 2nd row of sliced tensor
print("Result after sliced_tensor_example[1]: ", sliced_tensor_example[1])
print("Dimension after sliced_tensor_example[1]: ", sliced_tensor_example[1].ndimension())

#Lets try to get the values in row 2 and 3 in the second column. Note that the code below will not work.
print("Result: ", tensor_example[1:3][1])
tensor_example[1:3][1]
print("Dimension: ", tensor_example[1:3][1].ndimension()) #This gives dimension of 1

#In order to get the values in row 2 and 3 in the second column. we have to separate with a comma.
tensor_example[1:3, 1]


#Lets modify values in a 2D tensor.
tensor_ques = torch.tensor([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
#We will modify the values in the second column of the second and 3rd row. 
tensor_ques[1:3, 1] = 0
tensor_ques







