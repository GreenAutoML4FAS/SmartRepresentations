from utils import ROOT_DIR
from data.dataloader import Data
from data.cifar import CIFAR10, CIFAR100
from data.gtsrb import GTSRB
from data.fashion_mnist import FashionMNIST
from data.custom import Custom

datasets = [
    GTSRB(ROOT_DIR),
    CIFAR10(ROOT_DIR),
    CIFAR100(ROOT_DIR),
    FashionMNIST(ROOT_DIR),
    Custom(ROOT_DIR)
]
