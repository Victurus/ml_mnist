import os
import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch import no_grad
from torch import save as tsave

class Configuration():
    """ Class with configuration parameters """

    def __init__(self, num_epochs=5,num_classes=10,
            batch_size=100,learning_rate=0.001,
            data_path='./data',model_path='./model'):
        self.num_epochs    = num_epochs
        self.num_classes   = num_classes
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.data_path     = os.path.abspath(data_path)
        self.model_path    = os.path.abspath(model_path)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')

    def __str__(self):
        res = \
        "Number of epochs :{:<10}\n".format(self.num_epochs) + \
        "Number of classes:{:<10}\n".format(self.num_classes) + \
        "Batch size       :{:<10}\n".format(self.batch_size) + \
        "Learning_rate    :{:<10}\n".format(self.learning_rate) + \
        "Dataset path     :{:<10}\n".format(self.data_path) + \
        "Model path       :{:<10}\n".format(self.model_path) + \
        "Proceccing unit  :{:<10}".format(str(self.device).upper())
        return res

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential( nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Predefined Parameters
    conf = Configuration()
    print(conf)

    # Transformation and Dataset
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root=conf.data_path, train=True, transform=trans, download=True)
    test_dataset = datasets.MNIST(root=conf.data_path, train=False, transform=trans)

    # Data loaders, need to pass to learning
    train_loader = DataLoader(dataset=train_dataset, batch_size=conf.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=conf.batch_size, shuffle=False)

    # Model Training
    model = ConvNet()
    model.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    # Training
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    for epoch in range(conf.num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward calculation
            images = images.to(conf.device)
            labels = labels.to(conf.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Back propaganation and optimizer
            optimizer.zero_grad() # making gradient zero for further calculations
            loss.backward()
            optimizer.step()

            # Accuracy tracking
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            # Training Log
            if (i + 1) % conf.batch_size == 0:
                log = 'Epoch [{:3}/{:3}] | '.format(epoch + 1, conf.num_epochs)+\
                      'Step  [{:4}/{:4}] | '.format(i+1, total_step)+\
                      'Loss: {:7.4f} | '.format(loss.item())+\
                      'Accuracy: {:7.4f}%'.format((correct / total) * 100)
                print(log)

    model.eval()
    with no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(conf.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_dataset), (correct / total) * 100))

    tsave(model.state_dict(), conf.model_path + '/conv_net_model.ckpt')
