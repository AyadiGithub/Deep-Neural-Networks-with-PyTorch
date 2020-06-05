"""

Prebuilt Dataset 

"""
#Imports needed
import torch 
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)
from matplotlib.pyplot import imshow
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dsets

#Function needed for this exercise
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray') #Coverts to numpy and reshapes 
    plt.title('y = ' + data_sample[1]) #Plots title of the image


#Lets Import the prebuilt dataset into variable dataset
#We creat the dataset object
#Root is root directory of dataset, train parameter indicates if we want to use training or testing sets. 
#Download = True: downlaods the dataset into the directory. We set transform parameter to convert image to tensor.
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

#Each element of the dataset object contains a tuple. 
#Lets check the attributes of the elements
print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")

show_data(dataset[0]) #The first sample at index 0 is the img of number 7
show_data(dataset[1]) #The second sample at index 1 is the img of number 2


#Lets apply some combination of transforms on the dataset using Compose
#We will crop to 20x20 and transform to tensor
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape) #The shape is now 1x20x20 from 1x28x28

# Plot the first element in the dataset (The number 7)
show_data(dataset[0],shape = (20, 20)) #The new cropped img does not have as much black as before
# Plot the second element in the dataset (The number 2)
show_data(dataset[1],shape = (20, 20)) #The new cropped img does not have as much black as before

#Lets try another kind of transform combination
#Lets flip the image horizontally and convert it to a tensor
#We will have to transform the original source dataset
fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = fliptensor_data_transform)
show_data(dataset[1])
show_data(dataset[2])

#Lets do a vertical flip then a horizontal flip then transform to tensor
my_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p = 1), transforms.RandomHorizontalFlip(p = 1), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = my_data_transform)
show_data(dataset[1])
show_data(dataset[2])
