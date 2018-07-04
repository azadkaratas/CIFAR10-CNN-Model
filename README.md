# CIFAR10-CNN-Model
I built a Neural Network model with PyTorch library to recognize the CIFAR10 dataset, which consists of 10 different object images. Since PyTorch supports CIFAR10 dataset, accessing and loading of train and test images was easy. 

In the neural network model, I achieved 83% performance by using 5 convolutional layers followed by 2 fully connected layers.  Network is implemented with GPU accelerated PyTorch library. For fast training of the network, I used the fast computers supported by Google Colaboratory. It significantly reduces amount of training time. 

CNN architecture is shown below:

![alt text](https://github.com/azadkaratas/CIFAR10-CNN-Model/blob/master/image/model.png)

Another simple representation of the proposed model is as shown below:

Input Image  -> Conv (1x1x56) -> BN -> ReLU -> Conv (2x2x84) -> BN -> ReLU -> Conv (2x2x128)  -> BN -> ReLU -> Max Pool (2x2) -> Conv (2x2x256) -> BN -> ReLU -> Max Pool (2x2) -> Conv (2x2x512) -> BN -> ReLU -> Max Pool (2x2) -> Dropout -> FC1 -> FC2  -> CrossEntropyLoss

# Result

This designed neural network model gives 83% accuracy for all 10000 images in CIFAR10 dataset. Thanks to Google Colaboratory, I managed to tune my hyperparameters in a quick way. 
