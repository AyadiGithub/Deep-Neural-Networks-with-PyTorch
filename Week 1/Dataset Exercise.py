'''

Dataset Exercise

'''

#Imports needed
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(0)
from matplotlib.pyplot import imshow
from PIL import Image
import pandas as pd
import os

#Function needed for this exercise
def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray') #Coverts to numpy and reshapes 
    plt.title('y = ' + data_sample[1]) #Plots title of the image
    
# Read CSV file from the URL and print out the first five samples
directory = ""
csv_file ='index.csv'
csv_path = os.path.join(directory,csv_file)

data_name = pd.read_csv(csv_path)
data_name.head()

#The first column of the dataframe corresponds to the type of clothing. 
#The second column is the name of the image file corresponding to the clothing. 
data_name.iloc[0, 0] #Obtaining the class of first sample.
data_name.iloc[0, 1] #Obtaining the name of the first sample


#Lets load the images and assign their name and path
image_name = data_name.iloc[1, 1]
image_name

image_path = os.path.join(directory,image_name)
image_path

#Lets create a loop that loads all 100 images
for i in range (0,101): #Lets plot all the images in the imageset
    image_name = data_name.iloc[i,1] #we iterate through every image in the i'th row, 2nd column.
    image_path = os.path.join(directory,image_name) #We set the image path 
    image = Image.open(image_path) #We open the image from its path
    #Lets see the image
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(data_name.iloc[i,1])
    plt.show()


#Lets create a Dataset class
class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        
        # Image directory
        self.data_dir = data_dir
        
        # The transform is goint to be used on image
        self.transform = transform
        data_dircsv_file = os.path.join(self.data_dir,csv_file)
        # Load the CSV file contians image info
        self.data_name = pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0] 
    
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        
        # Image file path
        img_name = os.path.join(self.data_dir,self.data_name.iloc[idx, 1])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 0]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

#Creating a dataset object
dataset = Dataset(csv_file = csv_file, data_dir = directory)

#Lets check the image name and label and plot
image = dataset[0][0]
y = dataset[0][1]
plt.imshow(image,cmap='gray', vmin=0, vmax=255)
plt.title(y)
plt.show()


#Lets transform the images
#We need transforms from torchvision
import torchvision.transforms as transforms

#We can use the Compose method to combine transforms in the order we want
#We will transform 28x28 images to 20x20 tensor
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = Dataset(csv_file=csv_file , data_dir=directory,transform=croptensor_data_transform )
print("The shape of the first element tensor: ", dataset[0][0].shape)


#Plot the first element in the dataset
#We will use the function shwo_data that was created
show_data(dataset[1], shape = (20, 20))
show_data(dataset[2], shape = (20, 20))

#Lets try another combination of transforms
fliptensor_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p=1),transforms.ToTensor()])
dataset_flipped = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset_flipped[1])

#Lets try another transform combination
my_data_transform = transforms.Compose([transforms.RandomVerticalFlip(p = 1), transforms.RandomHorizontalFlip(p = 1), transforms.ToTensor()])
dataset_transformed = Dataset(csv_file=csv_file , data_dir=directory,transform=fliptensor_data_transform )
show_data(dataset_transformed[1])




















