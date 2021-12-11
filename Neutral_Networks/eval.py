import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import MLP, LeNet5
import utility as utility


def get_accuracy(loader,model):
    if loader.dataset.train:
        print('Getting accuracy on training data.')
    else:
        print('Getting accuracy on testing data.')
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_corrects = 0
    n_sample   = 0
    model.to(device)
    model.eval() 

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            if type(model).__name__ == 'MLP':
              x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _, y_pred = scores.max(1)
            n_corrects += (y_pred == y).sum()
            n_sample += y_pred.size(0)

        print(f'We got {n_corrects}/{n_sample} correct. Accuracy = {float(n_corrects)/float(n_sample)*100.0:.2f}')
    model.train()

if __name__ == '__main__':
    args = utility.parse_args()
    input_size = 784
    n_classes = 10
    model_path = ''
    model = MLP(input_size,n_classes)
    model.load_state_dict(torch.load(model_path))
    batch_size     = 64
    if args.dataset == 'FashionMNIST':
        if args.model == 'MLP':
            nodes = [784,100,100,50,25]
            model = MLP(nodes,n_classes)
            transform=transforms.ToTensor()
        elif args.model == 'LeNet5':
            first_channel = 1
            model = LeNet5(first_channel,n_classes)
            transform=transforms.Compose([transforms.resize((32,32)),transforms.ToTensor()])
        else:
            raise TypeError('Please choose model : MLP or LeNet5')
        train_dataset = datasets.FashionMNIST(root='datasets/fashion_mnist',train=True,transform=transforms,download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset  = datasets.FashionMNIST(root='datasets/fashion_mnist',train=False,transform=transforms,download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    elif args.dataset == 'CIFAR10':
        if args.model == 'MLP':
            nodes = [3072,1024,512,256,64]
            model = MLP(nodes,n_classes)
            transform=transforms.ToTensor()
        elif args.model == 'LeNet5':
            first_channel = 3
            model = LeNet5(first_channel,n_classes)
            transform=transforms.ToTensor()
        else:
            raise TypeError('Please choose model : MLP or LeNet5')
        train_dataset = datasets.CIFAR10(root='datasets/cifar10',train=True,transform=transforms,download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset  = datasets.CIFAR10(root='datasets/cifar10',train=False,transform=transforms,download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    else : print('Please choose dataset : FashionMNIST or CIFAR10')
    get_accuracy(train_loader,model)
    get_accuracy(test_loader,model)

