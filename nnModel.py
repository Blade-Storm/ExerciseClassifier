import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        '''
            Init function defines the layers for the neural network

            This network is created with three convolutional layers and two linear layers
        '''

        # 224 x 224 x 3
        self.conv1 = nn.Conv2d(3, 16, 3)
        # 74 x 74 x 16
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 24 x 24 x 32
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(3, 3)

        # 8 x 8 x 64
        self.fc3 = nn.Linear(3136, 1024) 
        self.output = nn.Linear(1024, 3)

    
    def forward(self, x):
        '''
            The forward method for the model

            input: 224 x 244 x 3 images
            output: Log of the ineference on the class
        '''
        x = self.pool((F.relu(self.conv1(x))))
        x = self.pool((F.relu(self.conv2(x))))
        x = self.pool((F.relu(self.conv3(x))))                

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.output(x), dim=1)
        return x

    def num_flat_features(self, x):
        '''
            Converts the output from the convolutional layer to the input size for the Linear layer

            input: The output from the previous convolutional layer
            output: The number of features for the size of the linear layer
        '''
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def initialize_weights(self, m):
        '''
            Initialize the weights using uniform normal distribution

            input: The model to initialize the weights for
        '''
        # Get the name of the classes in the model
        classname = m.__class__.__name__

        # For each Linear layer initialize the weights using uniform normal distribution
        if classname.find('Linear') != -1:
            # Get the number of input features
            n = m.in_features

            # Get the normal distribution centered at 0 
            y = np.random.normal(n)
            y = 1 / np.sqrt(y)

            # Initialize the weights uniformly and Bias with 0
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

