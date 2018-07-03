import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# PyTorch CIFAR10 Dataset loading and normalization process [-1, 1]
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Definition of a CNN Model with 5 conv. layers
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

net = Net()


# Definition of Stochastic Gradient Descent with Momentum as a Loss function and optimizer
init_lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0.9)

# Training of the network

# GPUs are running here
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(25):  # loop over the dataset multiple times
    running_loss = 0.0

    if epoch == 10:
      for param_group in optimizer.param_groups:
          param_group['lr'] = 0.0001
    elif epoch == 20:
      for param_group in optimizer.param_groups:
          param_group['lr'] = 0.00001
          
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) #GPU İŞLEMİ İÇİN GEREKLİ

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save(net, 'model.pkl')
