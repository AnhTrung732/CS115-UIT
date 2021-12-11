from random import shuffle
import torch
from torch import nn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import MLP, LeNet5
import utility as utility
from collections import OrderedDict
from eval import get_accuracy
import os
import time
import numpy as np


def copyStateDict(state_dict):
    start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict



def train(train_loader,test_loader,model,batch_size,lr,n_epochs,interval,pths_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        if (epoch) % 2 == 0 and epoch != 0:
          get_accuracy(train_loader,model)
          get_accuracy(test_loader,model)
        for bathch_idx, (data, targets) in enumerate(train_loader):
            # Get data to GPU
            data = data.to(device)
            targets = targets.to(device)
            if type(model).__name__ == 'MLP':
              data = data.reshape(data.shape[0], -1)

            score = model(data)
            loss = criterion(score,targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if (bathch_idx+1)%100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Batch {bathch_idx}, Loss: {loss.item():.2f}")
 
            if (epoch + 1) % interval == 0:
                state_dict = model.state_dict()
                save_path = os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch+1))
                torch.save(state_dict, save_path)

    return state_dict



if __name__ == '__main__':
    args = utility.parse_args()
    batch_size     = 64
    lr             = 1e-3
    n_epochs 	   = 50
    n_classes      = 10
    interval       = 10
    if args.dataset == 'FashionMNIST':
        if args.model == 'MLP':
            print('Training MLP model on FashionMNIST dataset')
            nodes = [784,100,100,50,25]
            model = MLP(nodes,n_classes)
            transform=transforms.ToTensor()
            pths_path      = 'model/fashion_mnist/mlp'
        elif args.model == 'LeNet5':
            print('Training LeNet5 model on FashionMNIST dataset')
            first_channel = 1
            model = LeNet5(first_channel,n_classes)
            transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
            pths_path      = 'model/fashion_mnist/lenet5'
        else:
            raise TypeError('Please choose model : MLP or LeNet5')
        train_dataset = datasets.FashionMNIST(root='datasets/fashion_mnist',train=True,transform=transform,download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset  = datasets.FashionMNIST(root='datasets/fashion_mnist',train=False,transform=transform,download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    elif args.dataset == 'CIFAR10':
        if args.model == 'MLP':
            print('Training CIFAR10 model on MLP dataset')
            nodes = [3072,1024,512,256,64]
            model = MLP(nodes,n_classes)
            transform=transforms.ToTensor()
            pths_path      = 'model/cifar10/mlp'
        elif args.model == 'LeNet5':
            print('Training CIFAR10 model on LeNet5 dataset')
            first_channel = 3
            model = LeNet5(first_channel,n_classes)
            transform=transforms.ToTensor()
            pths_path      = 'model/cifar10/lenet5'
        else:
            raise TypeError('Please choose model : MLP or LeNet5')
        train_dataset = datasets.CIFAR10(root='datasets/cifar10',train=True,transform=transform,download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset  = datasets.CIFAR10(root='datasets/cifar10',train=False,transform=transform,download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    else : raise TypeError('Please choose dataset : FashionMNIST or CIFAR10')
    
    train(train_loader,test_loader,model,batch_size,lr,n_epochs,interval,pths_path)	
    