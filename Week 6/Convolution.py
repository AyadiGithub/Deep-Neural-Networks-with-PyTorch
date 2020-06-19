"""

What is Convolution?
- Convolution is a linear operation similar to a linear equation, dot product, or matrix multiplication. 
- Convolution has several advantages for analyzing images.

In convolution, the parameter w is called a kernel. 
You can perform convolution on images where you let the variable image denote the variable X and w denote the parameter.

Max pooling simply takes the maximum value in each region.


"""

# In[1] Imports

import torch 
import torch.nn as nn

# In[2] Conv 2D

# in and out channels = 1 and Kernel = 3,3. Stride default(1,1)
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
conv

# 3x3 tensor for kernel weights
Gx = torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0,-1.0]])

# Lets give the conv2d manually assigned values
conv.state_dict()['weight'][0][0] = Gx
conv.state_dict()['bias'][0]=0.0
conv.state_dict()

# Create image
image=torch.zeros(1,1,5,5)
image[0,0,:,2] = 1
image

# Create relu function
relu = nn.ReLU()

# Apply conv to image and then Relu
Z = conv(image)
Z
Z = relu(Z)
Z

# Create a second image
image1=torch.zeros(1,1,4,4)
image1[0,0,0,:] = torch.tensor([1.0,2.0,3.0,-4.0])
image1[0,0,1,:] = torch.tensor([0.0,2.0,-3.0,0.0])
image1[0,0,2,:] = torch.tensor([0.0,2.0,3.0,1.0])
image1


# Create a maxpooling object in 2D
max1 = torch.nn.MaxPool2d(2, stride=1)

# Apply maxpooling to image
max1(image1)

# Create another maxpooling object without setting stride
# default stride is equal to kernel_size
max2 = torch.nn.MaxPool2d(2)

# Apply max2 to image
max2(image1)










