import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# The Beta Finder Network
class BFNet(nn.Module):
    def __init__(self, classes=2):
        super(BFNet, self).__init__()
        self.classes = 2
        # Convolution
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(10,2), stride=(3,1), padding=(0,1))
        self.relu1 = nn.ReLU(inplace=True)

        # Fully-connected for classification
        self.fc1 = nn.Conv2d(10, classes, kernel_size=(10,1))

        # ConvTranspose
        self.upscore1 = nn.ConvTranspose2d(classes, 10, kernel_size=(10,1))
        self.upscore2 = nn.ConvTranspose2d(10,
                                           1,
                                           kernel_size=(10,1),
                                           stride=(3,1),
                                           padding=(0,1),
                                           bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.upscore1(x)
        x = self.upscore2(x)
        return x



class BFModel(object):
    def __init__(self, num_breaks) -> None:
        super().__init__()
        self.num_breaks = num_breaks
        self.net = BFNet()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def train(self, X, Y):
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        X = X.reshape((1, 1, X.shape[0], X.shape[1]))
        Y = Y.reshape((1, 1, Y.shape[0], 1))
        X = F.interpolate(X, size=(10000, 2))
        Y = F.interpolate(X, size=(10000, 1))
        print(X.shape)
        for epoch in range(30):
            inputs, labels = X, Y
            ypred = self.net.forward(inputs)
            # print(ypred.max())
            loss = self.loss(ypred, labels)
            loss.backward()
            self.optimizer.step()
            print("epoch: {} Loss:{}".format(epoch, loss),end="\n")
        print("Finished trainning...")
        torch.save(self.net.state_dict(), './bfnet.model')

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath))

    def predict(self, X):
        X = X.reshape((1, 1, X.shape[0], X.shape[1]))
        X = torch.Tensor(X)
        output = self.net.forward(X)
        return output.detach().numpy()
