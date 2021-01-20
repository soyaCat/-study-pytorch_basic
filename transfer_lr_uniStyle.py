import torch
import torch.nn as nn
import numpy as np

import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

resnet = models.resnet50(pretrained = True)

batch_size = 2
lr = 0.001
totalEpoch = 10
num_category = 2
img_dir = "./transfer_data/images/"

class Resnet_model(nn.Module):
    def __init__(self):
        super(Resnet_model, self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:-1])
        self.layer1 = nn.Sequential(
            nn.Linear(2048, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, num_category),
        )

    def forward(self, x):
        out = self.layer0(x)
        out = out.view(batch_size, -1)
        out = self.layer1(out)
        return out

class Resnet():
    def __init__(self, device):
        self.batch_size = batch_size
        self.device = device
        self.model = Resnet_model().to(self.device)
        for params in self.model.layer0.parameters():
            params.require_grad = False
        for params in self.model.layer1.parameters():
            params.require_grad = True
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.layer1.parameters(), lr = lr)

    def load_data(self, img_dir):
        img_dir = img_dir
        img_data = dset.ImageFolder(img_dir, transforms.Compose([
                                    transforms.RandomSizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    ]))
        print(img_data.classes)
        print(img_data.class_to_idx)
        print(img_data.imgs)

        train_loader = DataLoader(img_data, batch_size = self.batch_size,
                                    shuffle = True, num_workers = 2, drop_last = True)

        for img, label in train_loader:
            print(img.size())
            print(label)
        
        print('\n')
        return train_loader

    def train_model(self, train_loader):
        for index, [image, label] in enumerate(train_loader):
            x = image.to(self.device)
            y_ = label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model.forward(x)
            loss = self.loss_func(output, y_)
            loss.backward()
            self.optimizer.step()

            return loss

    def test_model(self, test_loader):
        test_loader = test_loader
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for image, label in train_loader:
                x = image.to(self.device)
                y_ = label.to(self.device)

                output = self.model.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct +=(output_index == y_).sum().float()
        
        return correct, total



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Resnet = Resnet(device)
    train_loader = Resnet.load_data(img_dir)

    for epoch in range(totalEpoch):
        loss = Resnet.train_model(train_loader)
        if epoch % 10 == 0:
            print(loss)

    correct = 0
    total = 0

    correct, total = Resnet.test_model(train_loader)
    print("Accuracy of Train Data: {}".format(100*correct/total))