"""

Convolution 2D with Multiple Outputs

"""

# In[1] Imports

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# In[2] Conv 2D

# Conv2D with in_channels = 1 and out_channels = 3 and Kernel = 3,3. Stride default(1,1)
conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
conv1

# Creating tensors for kernel weights
Gx = torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
Gy = torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
Gz = torch.ones(3,3)

# Manual initialization of kernel weights and bias
conv1.state_dict() # dictionary containing keys (weights, bias)
conv1.state_dict()['weight'][0][0] = Gx 
conv1.state_dict()['weight'][1][0] = Gy
conv1.state_dict()['weight'][2][0] = Gz
conv1.state_dict()['weight']
conv1.state_dict()['bias'][:] = torch.tensor([0.0,0.0,0.0])
conv1.state_dict()['bias']

# Showing kernels weights
for x in conv1.state_dict()['weight']:
    print(x)

# Showing bias in kernels
for x in conv1.state_dict()['bias']:
    print(x)
    
# In[3] Create Data/Image

image = torch.zeros(1, 1, 5, 5) # Vertical white block in the middle
image[0,0,:,2] = 1
image 

image1 = torch.zeros(1, 1, 5, 5) # Horizontal white block in the middle
image1[0, 0, 2, :] = 1
image1

# plot image
plt.imshow(image[0, 0, :, :].numpy(), cmap = 'gray')
plt.colorbar()
plt.show

plt.imshow(image1[0, 0, :, :].numpy(), cmap = 'gray')
plt.colorbar()
plt.show


# In[4] Perform convolution and Plot

out = conv1(image)
out1 = conv1(image1)

out.shape
out1.shape

# Print channels as tensors/image and plot for image
for channel, image in enumerate(out[0]):
    plt.imshow(image.detach().numpy(), cmap = 'gray')
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# Print channels as tensors/image and plot for image1
for channel, image1 in enumerate(out1[0]):
    plt.imshow(image1.detach().numpy(), cmap = 'gray')
    print(image1)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# It can be seen that different kernels can be used to detect various features in an image. 


# In[4] Create Data

image2 = torch.zeros(1,2,5,5)
image2[0,0,2,:] = -2
image2[0,1,2,:] = 1
image2
image2.shape

for channel,image in enumerate(image2[0]):
    plt.imshow(image.detach().numpy(), cmap = 'gray')
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()
    

# In[5] Create Conv2D object

# Conv2D object with 2 inputs and 1 ouput, kernel (3,3)

conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

# Manual kernel initialization

Gx1 = torch.tensor([[0.0,0.0,0.0],[0,1.0,0],[0.0,0.0,0.0]])
conv3.state_dict()['weight'][0][0] = 1*Gx1
conv3.state_dict()['weight'][0][1] = -2*Gx1
conv3.state_dict()['bias'][:] = torch.tensor([0.0])
conv3.state_dict()
conv3.state_dict()['weight']

# In[6] Perform convolution and Plot

out3 = conv3(image2)

# Print channels as tensors/image and plot for image1
for channel, image2 in enumerate(out3[0]):
    plt.imshow(image2.detach().numpy(), cmap = 'gray')
    print(image1)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()


# In[7] Create Conv2D and Data with multiple inputs and outputs

# Create conv2D with 2 in_channels and 3 out_channels, kernel_size (3,3)
conv4 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3)

# Manual weight and bias initialization
conv4.state_dict()['weight'][0][0] = torch.tensor([[0.0,0.0,0.0],[0,0.5,0],[0.0,0.0,0.0]])
conv4.state_dict()['weight'][0][1] = torch.tensor([[0.0,0.0,0.0],[0,0.5,0],[0.0,0.0,0.0]])

conv4.state_dict()['weight'][1][0] = torch.tensor([[0.0,0.0,0.0],[0,1,0],[0.0,0.0,0.0]])
conv4.state_dict()['weight'][1][1] = torch.tensor([[0.0,0.0,0.0],[0,-1,0],[0.0,0.0,0.0]])

conv4.state_dict()['weight'][2][0] = torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv4.state_dict()['weight'][2][1] = torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])

conv4.state_dict()['bias'][:]=torch.tensor([0.0,0.0,0.0])


image4 = torch.zeros(1,2,5,5)

image4[0][0] = torch.ones(5,5)

image4[0][1][2][2] = 1

for channel, image in enumerate(image4[0]):
    plt.imshow(image.detach().numpy(), cmap = 'gray' )
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()


# Apply conv to image
z = conv4(image4)
z

