import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 784  # size of pic
num_classes = 10
num_epochs = 10
batch_size = 50
learning_rate = 0.001

train_dataset = dsets.MNIST()
