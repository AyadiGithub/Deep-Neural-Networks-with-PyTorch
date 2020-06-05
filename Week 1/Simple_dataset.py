"""

Simple Dataset 

"""

#To create a dataset class in torch.
import torch
from torch.utils.data import Dataset #for dataset creation in torch
#The torch.manual_seed() is for forcing the random function to give the same number every time we try to recompile it.
torch.manual_seed(1)

#We create out own dataset toy_set
class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len

#let us create our toy_set object, find out the value on index 1 and the length of the inital dataset
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))

#We are able to customize the indexing and length method by def __getitem__(self, index) and def __len__(self).

#Print out the first 3 elements and assign them to x and y. We use a for loop for this.

for i in range(3): #This loop will iterate through indexes 0,1,2 and fetches the x, y value for each index.
    x, y = our_dataset[i]
    print("index", i, "x:", x, "y", y)


for i in range(len(our_dataset)): #This loop prints all the x,y values in the dataset with each index numbered.
    print(i) 
    print(' x:', x, 'y:', y) 

#Another way to do this is as follows:
for x,y in our_dataset:
    print(' x:', x, 'y:', y)
    
    
#Lets create a new dataset with length 50
my_new_dataset = toy_set(length=50)
print("My new dataset length: ", len(my_new_dataset))
#Lets print all the values.
for i in range(len(my_new_dataset)): #This loop prints all the x,y values in the dataset with each index numbered.
    print(i) 
    print(' x:', x, 'y:', y) 



"""
Lets create a class to transform data.

"""

class add_mult(object): #We create a transform class
    
    #Constructor
    def __init__(self, addx = 1, muly = 2):
        self.addx = addx
        self.muly = muly
    
    #Executor
    def __call__(self, sample): #
        x = sample[0]
        y = sample[1]
        x = x + self.addx #Transform will add 1 to x
        y = y * self.muly #Transform will multiply y by 2
        sample = x, y
        return sample

#Now lets create a transform object
a_m = add_mult()
data_set = toy_set()

#Lets Assign the outputs of the original dataset to x and y. 
#And apply the transform add_mult to the dataset and output the values as x_ and y_, respectively: 
#Lets iterate over the first 10 elements and show the original x and y and tranformed x and y
for i in range(10):
    x, y = data_set[i]
    print('Index : ', i, 'Original x: ', x, 'Original y: ', y)
    x_,y_ = a_m(data_set[i])
    print('Index : ', i, 'Transformed x: ', x_, 'Transformed y: ', y_)



"""

Composing multiple transform on data.

"""

#You can compose multiple transforms on the dataset object. 
#We need to import transforms from torchvision

from torchvision import transforms

#Lets create a new transform class

class mult(object):
    
    # Constructor
    def __init__(self, mult = 100):
        self.mult = mult
        
    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample



#Now we have 2 transform classes and we can use Compose to combine them
data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose: ", data_transform)
data_transform(data_set[0]) #Lets apply our combined transform to index 0 of data_set and see it
x,y = data_transform(data_set[0])

print( 'Original x: ', x, 'Original y: ', y)
print( 'Transformed x_:', x_, 'Transformed y_:', y_)

#Now we can pass the new Compose object (The combination of methods add_mult() and mult) to the constructor for creating toy_set object.
#Create a new toy_set object with compose object as transform
compose_data_set = toy_set(transform = data_transform)

#Lets print out the elements in our new transformed dataset
for i in range(len(compose_data_set)): 
    print('Index: ', i, ' x: ', x, 'y: ', y)     


#Lets reverse the order of the transforms
my_compose = transforms.Compose([mult(), add_mult()])
my_transformed_dataset = toy_set(transform = my_compose)
for i in range(len(my_transformed_dataset)):
    print('Index: ', i, 'Transformed x:', x_, 'Transformed y:', y_)





