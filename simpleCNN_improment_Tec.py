import torch
import torch.nn as nn
import numpy
import torch.optim.lr_scheduler as lr_scheduler
#import 

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 256
lr = 0.0002
weight_decay = 0.00002
lr_dacay = 0.90
totalEpochs = 10

mnist_train = dset.MNIST("./", train = True, transform= transforms.ToTensor(),
                        target_transform=None, download=False)
mnist_test = dset.MNIST("./", train = False, transform=transforms.ToTensor(),
                        target_transform=None, download= False)
CIFAR10_train = dset.CIFAR10("./", train = True, transform= transforms.ToTensor(),
                            target_transform=None, download=False)
CIFAR10_test = dset.CIFAR10("./", train = False, transform=transforms.ToTensor(),
                            target_transform=None, download= False)
train_loader = torch.utils.data.DataLoader(CIFAR10_train, batch_size = batch_size, 
                                            shuffle = True, num_workers=2, drop_last= True)
test_loader = torch.utils.data.DataLoader(CIFAR10_test, batch_size = batch_size,
                                            shuffle= False, num_workers= 2, drop_last= True)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

class CNNModel_with_init(nn.Module):
    def __init__(self):
        super(CNNModel_with_init, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                '''
                #Init wiht small numbers
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                use Kaming_normal if use ReLU()
                '''
                #Xavier Initialization
                #nn.init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
                
                #kaming Initialization
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.Linear):
                #Xavier Initalization
                #nn.init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
                #kaming Initialization
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

class CNNModel_with_batchnorm(nn.Module):
    def __init__(self):
        super(CNNModel_with_batchnorm, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

class CNN():
    def __init__(self, device):
        self.device = device
        self.model = CNNModel().to(device)
        self.model_with_init = CNNModel_with_init().to(device)
        self.model_with_batchnorm = CNNModel_with_batchnorm().to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sceduler = lr_scheduler.StepLR(self.optimizer, step_size = 1, gamma= lr_dacay)
        self.L2optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.optimizer_with_init = torch.optim.Adam(self.model_with_init.parameters(), lr = lr)
        self.optimizer_with_batchnorm = torch.optim.Adam(self.model_with_batchnorm.parameters(), lr = lr)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, loss_arr):
        for j, [image, label] in enumerate(self.train_loader):
            x = image.to(self.device)
            y_ = label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model.forward(x)
            loss = self.loss_func(output, y_)
            loss.backward()
            self.optimizer.step()

            if j % 1000 == 0:
                #print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
                self.test_model()
                '''
                Accuracy of Test Data: 9.985978126525879
                Accuracy of Test Data: 34.91586685180664
                Accuracy of Test Data: 39.933895111083984
                Accuracy of Test Data: 44.64142608642578
                Accuracy of Test Data: 46.90504837036133
                Accuracy of Test Data: 46.80488967895508
                Accuracy of Test Data: 49.609375
                Accuracy of Test Data: 50.060096740722656
                Accuracy of Test Data: 51.302085876464844
                Accuracy of Test Data: 52.544071197509766
                '''
                '''
                with scheduler....
                Accuracy of Test Data: 11.698718070983887
                0 [0.0018000000000000002]
                Accuracy of Test Data: 46.36418151855469
                1 [0.0016200000000000001]
                Accuracy of Test Data: 53.846153259277344
                2 [0.001458]
                Accuracy of Test Data: 56.78084945678711
                3 [0.0013122000000000001]
                Accuracy of Test Data: 61.037662506103516
                4 [0.00118098]
                Accuracy of Test Data: 62.24959945678711
                5 [0.001062882]
                Accuracy of Test Data: 62.57011413574219
                6 [0.0009565938]
                Accuracy of Test Data: 65.31450653076172
                7 [0.00086093442]
                Accuracy of Test Data: 66.22595977783203
                8 [0.000774840978]
                Accuracy of Test Data: 66.71675109863281
                9 [0.0006973568802]
                '''
    
    def train_model_with_L2optim(self, loss_arr):
        for j, [image, label] in enumerate(self.train_loader):
            x = image.to(self.device)
            y_ = label.to(self.device)

            self.L2optimizer.zero_grad()
            output = self.model.forward(x)
            loss = self.loss_func(output, y_)
            loss.backward()
            self.L2optimizer.step()

            if j % 1000 == 0:
                #print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
                self.test_model()
                '''
                Accuracy of Test Data: 10.036057472229004
                Accuracy of Test Data: 34.05448913574219
                Accuracy of Test Data: 40.10416793823242
                Accuracy of Test Data: 44.481170654296875
                Accuracy of Test Data: 46.23397445678711
                Accuracy of Test Data: 47.756412506103516
                Accuracy of Test Data: 48.667869567871094
                Accuracy of Test Data: 50.050079345703125
                Accuracy of Test Data: 50.090145111083984
                Accuracy of Test Data: 51.78285217285156
                '''

    def train_model_with_init(self, loss_arr):
        for j, [image, label] in enumerate(self.train_loader):
            x = image.to(self.device)
            y_ = label.to(self.device)

            self.optimizer_with_init.zero_grad()
            output = self.model_with_init.forward(x)
            loss = self.loss_func(output, y_)
            loss.backward()
            self.optimizer_with_init.step()

            if j % 1000 == 0:
                #print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
                self.test_model_with_init()
                '''
                Accuracy of Test Data: 10.006010055541992
                Accuracy of Test Data: 42.427886962890625
                Accuracy of Test Data: 49.0384635925293
                Accuracy of Test Data: 51.68269348144531
                Accuracy of Test Data: 53.65584945678711
                Accuracy of Test Data: 55.949520111083984
                Accuracy of Test Data: 56.40024185180664
                Accuracy of Test Data: 56.209938049316406
                Accuracy of Test Data: 57.74238967895508
                Accuracy of Test Data: 58.95432662963867
                '''
    
    def train_model_with_batchnorm(self, loss_arr):
        for j, [image, label] in enumerate(self.train_loader):
            x = image.to(self.device)
            y_ = label.to(self.device)

            self.model_with_batchnorm.train()
            self.optimizer_with_batchnorm.zero_grad()
            output = self.model_with_batchnorm.forward(x)
            loss = self.loss_func(output, y_)
            loss.backward()
            self.optimizer_with_batchnorm.step()

            if j % 1000 == 0:
                #print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
                self.test_model_with_batchnorm()
                '''
                Accuracy of Test Data: 10.006010055541992
                Accuracy of Test Data: 54.457130432128906
                Accuracy of Test Data: 60.897438049316406
                Accuracy of Test Data: 64.27283477783203
                Accuracy of Test Data: 66.23597717285156
                Accuracy of Test Data: 65.625
                Accuracy of Test Data: 67.5380630493164
                Accuracy of Test Data: 69.15064239501953
                Accuracy of Test Data: 68.06890869140625
                Accuracy of Test Data: 66.38621520996094
                '''

    def test_model_with_batchnorm(self):
        self.model_with_batchnorm.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in self.test_loader:
                x = image.to(self.device)
                y_ = label.to(self.device)

                output = self.model_with_batchnorm.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()
            print("Accuracy of Test Data: {}".format(100*correct/total))

    def test_model_with_init(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in self.test_loader:
                x = image.to(self.device)
                y_ = label.to(self.device)

                output = self.model_with_init.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()
            print("Accuracy of Test Data: {}".format(100*correct/total))

    def test_model(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for image, label in self.test_loader:
                x = image.to(self.device)
                y_ = label.to(self.device)

                output = self.model.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()
            print("Accuracy of Test Data: {}".format(100*correct/total))


if __name__ == '__main__':
    #freeze_support()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN(device)
    loss_arr = []
    for epoch in range(totalEpochs):
        cnn.train_model_with_batchnorm(loss_arr)
        #cnn.sceduler.step()
        #print(epoch, cnn.sceduler.get_last_lr())