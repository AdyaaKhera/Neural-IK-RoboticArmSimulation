#this file will contain the MLP for inverse kinematics

import torch, csv 
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader

#loading the dataset using arrays
x, y = [], []
with open("data/dataset.csv", newline = "") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        xval, yval = float(row[0]), float(row[1])
        theta1, theta2 = float(row[2]), float(row[3])
        x.append([xval, yval])
        y.append([theta1, theta2])

#converting them into tensors
x = torch.tensor(x, dtype = torch.float32)
y = torch.tensor(y, dtype = torch.float32)

""" 
print(x.shape, y.shape) #check
"""

#normalizing coordinate values - zscore
xmean, xstd = x.mean(dim = 0), x.std(dim = 0)
x = (x - xmean) / xstd

#normalizing angle values - min max
ymin = y.min(dim = 0).values
ymax = y.max(dim = 0).values
y = (y - ymin) / (ymax - ymin)

"""
#checking if the values are close to 0 and 1 or not
print("x mean/std:", x.mean(0), x.std(0))
print("y min/max:", y.min(0).values, y.max(0).values)
"""

dataset = TensorDataset(x, y) #dataset[i] is now (x_i, y_i)

train_size = int(0.8 * len(dataset)) #using 80 percent of the data to train the model
test_size = len(dataset) - train_size #using the leftover 20% to test it later on

train_set, test_set = random_split(dataset, [train_size, test_size]) #randomly generating distributing the dataset as 80-20

#feeding data in small chunks for better processing and accuracy
train_loader = DataLoader(train_set, batch_size = 64, shuffle = True) #this will load the data from the train set in batches of 64 after shuffling them
test_loader = DataLoader(test_set, batch_size = 64)

#building the MLP model

class InverseKinematicsModel(nn.Module):
    def __init__(self):
        super().__init__() #inversekinematics class inherits from nnmodule
        self.model = nn.Sequential(
            nn.Linear(2,64), #increasing size
            nn.ReLU(), #non-linearity
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 64), #reducing size
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self, x):
        return self.model(x)
    
model = InverseKinematicsModel() #creating an instance

"""
print(model) #check
"""

criterion = nn.MSELoss() #mean squared error loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #using the Adam optimizer
#the optimizer will adjust the parameters of the model and it's learning rate is 0.001

#training the model 100 times
for epoch in range(100):
    total_loss = 0
    for xbatch, ybatch in train_loader: #each batch will have 64 samples
        output = model(xbatch)
        loss = criterion(output, ybatch)
        optimizer.zero_grad() #crealing up gradients from last batch
        loss.backward()
        optimizer.step() #applying the changes
        total_loss += loss.item() #.item mconverts the tensor into a number
    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} — Avg Loss: {avg_loss:.4f}")

#test loop
model.eval() #set to evaluation mode
test_loss = 0

with torch.no_grad(): #diabled gradients
    for xbatch, ybatch in test_loader:
        output = model(xbatch)
        loss = criterion(output, ybatch)
        test_loss += loss.item() #converting tensor to number

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")