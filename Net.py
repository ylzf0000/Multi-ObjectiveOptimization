import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim


class Net(nn.Module):
    def __init__(self, x_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(3 * x_dim, 8),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(8, x_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(net, epochs, inputs, labels):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss))
        running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    pass
    # net = Net()
    # print(net)
    # net.zero_grad()
    # input = torch.randn(100, 2)
    # output = net(input)
    # criterion = nn.MSELoss()
    # import torch.optim as optim
    #
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # loss = criterion(output, target)
    #
    # loss.backward()
    # out.backward(torch.randn(100, 1))
    # print(out)
