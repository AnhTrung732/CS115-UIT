import argparse

def init_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("n_classes", type=int, default=10)
    # parser.add_argument("batch_size", type=int, default=64)
    # parser.add_argument("n_epochs", type=int, default=50)
    # parser.add_argument("learning_rate", type=int, default=1e-3)
    parser.add_argument("--model", type=str, default='MLP', help='Include MLP and LeNet')
    parser.add_argument("--dataset", type=str, default='FashionMNIST', help='Include FashionMNIST and CIFAR10' )
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()

if __name__ == '__main__':
    pass