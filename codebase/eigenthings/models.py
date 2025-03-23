import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset

from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def get_norm(planes, type):
    if type == 'BN':
        return nn.BatchNorm2d(planes)
    elif type == 'GN':
        return nn.GroupNorm(2, planes)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm_type, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = get_norm(planes, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = get_norm(planes, norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     get_norm(self.expansion * planes, norm_type)
                )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, lr, num_classes, norm_type, device, block=BasicBlock, num_blocks=[3, 3, 3]):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.device = device
        
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = get_norm(16, norm_type)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], norm_type, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], norm_type, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], norm_type, stride=2)
        self.classifier = nn.Linear(64, num_classes)

        self.apply(_weights_init)
        
        self.size = self.model_size()

    def _make_layer(self, block, planes, num_blocks, norm_type, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, norm_type, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


class ClientModel(nn.Module):
    def __init__(self, num_classes):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, self.num_classes)
        )

        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size


def cnn(num_classes=10):
    return ClientModel(num_classes)


class LSTM(nn.Module):
    def __init__(self, num_classes, embed_size=8, hidden_size=100):
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(num_classes, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size,hidden_size=hidden_size,num_layers=2)
        self.linear = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, (h_, c_) = self.lstm(x)
        last_hidden = output[-1,:,:]
        x = self.linear(last_hidden)
        return x


def lstm(num_classes=80):
    return LSTM(num_classes)


class ShakespeareDS(Dataset):
    def __init__(self, data, train=True):
        super(ShakespeareDS).__init__()
        self.tokens = []
        self.labels = []
        
        if train:
            for u in data["users_data"].values():
                self.tokens.extend(u["x"])
                self.labels.extend(u["y"])
        else:
            self.tokens = data["users_data"]["100"]["x"]
            self.labels = data["users_data"]["100"]["y"]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return torch.tensor(self.tokens[index]), torch.tensor(self.labels[index])