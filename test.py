import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


# PyTorch CIFAR10 Dataset loading and normalization process 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):    
    def __init__(self):
          super(Net, self).__init__()
          # 1 input image channel, 6 output channels, 5x5 square convolution
          # kernel
          self.conv1 = nn.Conv2d(3, 56, 1)
          self.conv1_bn = nn.BatchNorm2d(56)
          self.conv2 = nn.Conv2d(56, 84, 2)
          self.conv2_bn = nn.BatchNorm2d(84)
          self.conv3 = nn.Conv2d(84, 128, 2)
          self.conv3_bn = nn.BatchNorm2d(128)
          self.conv4 = nn.Conv2d(128, 256, 2)
          self.conv4_bn = nn.BatchNorm2d(256)
          self.conv5 = nn.Conv2d(256, 512, 2)
          self.conv5_bn = nn.BatchNorm2d(512)
          self.drop = nn.Dropout2d(p=0.2)
          
          self.fc1 = nn.Linear(4608, 2000)          
          self.fc2 = nn.Linear(2000, 10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x))) #Conv -> BN -> ReLu
        x = F.relu(self.conv2_bn(self.conv2(x))) #Conv -> BN -> ReLu
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))),2) #Conv -> BN -> ReLu -> Max Pooling
        x = F.max_pool2d(F.relu(self.conv4_bn(self.conv4(x))),2) #Conv -> BN -> ReLu -> Max Pooling 
        x = F.max_pool2d(F.relu(self.conv5_bn(self.conv5(x))),2) #Conv -> BN -> ReLu -> Max Pooling

        x = self.drop(x)
        
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#######################################################################
################  THIS IS WHERE THE MODEL IS LOADING   ################
################                                       ################
    
#NN_Model = torch.load('model.pkl', map_location='cpu') #.pkl model for CPU computers
NN_Model = torch.load('model.pkl')                      #.pkl model for GPU computers

################                                       ################
################                                       ################
#######################################################################


# Test the network on the test data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN_Model.to(device)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = NN_Model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#Show score for each class

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = NN_Model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))

