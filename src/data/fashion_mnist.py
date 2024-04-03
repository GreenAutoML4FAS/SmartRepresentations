import os
import numpy as np
import pandas as pd
from os.path import join, exists
from torchvision.datasets import FashionMNIST as fmnist

from data.dataloader import Data
from utils import ASSET_PATH

classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


class FashionMNIST(Data):

    def __init__(self, root):
        super().__init__(root=root)
        self.fmnist = None
        self.labels = list()

        self.properties = pd.DataFrame({
            "Property": [
                "Name",
                "Description",
                "Authors",
                "Number of Classes",
                "URL",
                "Date Published",
            ],
            "Value": [
                "GTSRB",
                "German Traffic Sign Recognition Benchmark (GTSRB)",
                "J. Stallkamp and M. Schlipsing and J. Salmen and C. Igel.",
                43,
                "https://benchmark.ini.rub.de/",
                2011,
            ]
        })
        self.thumbnail = join(ASSET_PATH, "images", "datasets", "FashionMNIST.png")

        if self.downloaded:
            self.load_data()

    def load_data(self):
        self.fmnist = fmnist(root=self.root, download=True)
        self.labels = self.fmnist.targets.numpy()

        self.data["idx"] = range(len(self.labels))
        self.data["label"] = self.labels
        self.data["class_name"] = \
            self.data["label"].apply(lambda x: classes[x])
        self.data["select"] = [False] * len(self.labels)

    def get_image(self, index):
        assert index < len(self), f"Index {index} out of range!"
        img, label = self.fmnist.__getitem__(index)
        img = np.array(img)
        return img

    def is_downloaded(self):
        if not exists(join(
                self.root, "FashionMNIST", "raw", "t10k-images-idx3-ubyte"
        )):
            return False
        return True

    def download(self):
        if not self.is_downloaded():
            prepare_data(self.root)
        self.load_data()


def prepare_data(root):
    d = fmnist(root=root, download=True)
    print(d.__dict__.keys())
    print(d.targets)
