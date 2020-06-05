"""

Dataset for Images

"""

import torch
from PIL import Image
import pandas as pd
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#We will use MNIST dataset, 28x28x1 images
#The dataset has 10 labels or classes

directory = r"C:\*******\Deep Neural Networks with PyTorch\Week 1"
csv_file = 'index.csv'
csv_path = os.path.join(directory,csv_file)

data_name = pd.read_csv((csv_path))
data_name.head() #Lets view the dataframe

print('File name:', data_name.iloc[0,1])
print('class or y:', data_name.iloc[0,0])

for i in range(len(data_name)): #Lets plot all the images in the imageset
    image_name = data_name.iloc[i,1] #we iterate through every image in the i'th row, 2nd column.
    image_path = os.path.join(directory,image_name) #We set the image path 
    image = Image.open(image_path) #We open the image from its path
    #Lets see the image
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(data_name.iloc[i,1])
    plt.show()
    

#Lets create a dataset class

class Dataset(Dataset):
    def __init__(self, csv_file, data_dir, transform = None):
        
        self.transform = transform
        self.data_dir = data_dir
        data_dircsv_file = os.path.join(self.data_dir,csv_file)
        self.data_name = pd.read_csv(data_dircsv_file)
        self.len = self.data_name.shape[0]
        
    def __lef__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        img_name = os.path.join(self.data_dir, self.data_name.iloc[idx, 1])
        image = Image.open(img_name)
        
        y = self.data_name.iloc[idx, 0]
        
        if self.transform:
            image = self.transform(image)
            return image, y


#Lets use torchvision pre-built transforms for images
transforms.CenterCrop(20) #Lets crop the image to 20x20
transforms.ToTensor() #Converting the image to a tensor

#Or we can combine the transforms
#Lets compose the transforms
croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])

#Now we apply the composed transforms to the dataset constructor
dataset = Dataset(csv_file = csv_file, data_dir = directory, transform = croptensor_data_transform)
dataset[0][0].shape #The dataset is 20x20x1



#Lets try a prebuilt dataset from torchvision
import torchvision.datasets as dsets

#We creat the dataset object
#Root is root directory of dataset, train parameter indicates if we want to use training or testing sets. 
#Download = True: downlaods the dataset into the directory. We set transform parameter to convert image to tensor.
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor)





