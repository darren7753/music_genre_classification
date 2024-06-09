"""
Defines the neural network, loss function and metrics

Architecture modified from the provided starter code

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, params):
        """
        Args:
            params: (Params) contains num_channels, num_classes
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.num_classes  = params.num_classes
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
    
        self.conv1 = nn.Conv2d(1, self.num_channels, 7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)


        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)

        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 7, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        self.conv4 = nn.Conv2d(self.num_channels*4, self.num_channels*8, 7, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.num_channels*8)

        self.conv5 = nn.Conv2d(self.num_channels*8, self.num_channels*16, 5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(self.num_channels*16)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(1*1*self.num_channels*16, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, self.num_classes)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 128 x 128 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        """
        #                                                  -> batch_size x 1 x 128 x 128
        # Apply the convolution layers, followed by batch normalisation, average pool and relu x 5

        s = self.bn1(self.conv1(s))                         # batch_size x num_channels*1 x 124 x 124
        s = F.relu(F.avg_pool2d(s, 2))                      # batch_size x num_channels*1 x 62 x 62

        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 58 x 58
        s = F.relu(F.avg_pool2d(s, 2))                      # batch_size x num_channels*2 x 29 x 29

        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 25 x 25
        s = F.relu(F.avg_pool2d(s, 2))                      # batch_size x num_channels*4 x 12 x 12

        s = self.bn4(self.conv4(s))                         # batch_size x num_channels*8 x  8 x  8
        s = F.relu(F.avg_pool2d(s, 2))                      # batch_size x num_channels*8 x  4 x  4

        s = self.bn5(self.conv5(s))                         # batch_size x num_channels*16 x  2 x  2
        s = F.relu(F.avg_pool2d(s, 2))                      # batch_size x num_channels*16 x  1 x  1


        # flatten the output for each image
        s = s.view(-1, 1*1*self.num_channels*16)             # batch_size x 1*1*num_channels*16

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x num_classes

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5, 6, 7]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
}
