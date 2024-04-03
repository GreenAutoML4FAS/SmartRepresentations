import os
import numpy as np
import pandas as pd
import pickle
from os.path import join, exists, basename
import urllib.request

from data.dataloader import Data
from utils import ASSET_PATH


class CIFAR10(Data):
    fine = True  # To be compatible to CIFAR100

    def __init__(self, root):
        super().__init__(root=root)

        self.images = list()
        self.labels = list()

        self.properties = pd.DataFrame({
            "Property": [
                "Name",
                "Description",
                "Number of Classes",
                "URL",
                "Date Published",
            ],
            "Value": [
                "CIFAR-10",
                "The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 "
                "million tiny images dataset. They were collected by "
                "Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.",
                10,
                "https://www.cs.toronto.edu/~kriz/cifar.html",
                2009,
            ]
        })
        self.thumbnail = join(ASSET_PATH, "images", "datasets", "CIFAR.png")

        if self.downloaded:
            self.load_data()

    def load_data(self):
        self.images = list()
        self.labels = list()
        batches = [
            "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
            "data_batch_5"
        ]
        for b in batches:
            with open(join(self.root, "cifar-10-batches-py", b), 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            self.images.append(np.reshape(data[b"data"], (10000, 3, 32, 32)))
            self.labels += data[b"labels"]
        self.images = np.concatenate(self.images, axis=0)
        self.images = self.images.transpose((0, 2, 3, 1))
        self.labels = np.asarray(self.labels)

        with open(join(self.root, "cifar-10-batches-py", "batches.meta"),
                  'rb') as fo:
            classes = pickle.load(fo, encoding='bytes')
        self.classes = [x.decode() for x in classes[b'label_names']]

        self.data["idx"] = range(len(self.labels))
        self.data["label"] = self.labels
        self.data["class_name"] = \
            self.data["label"].apply(lambda x: self.classes[x])
        self.data["select"] = [False] * len(self.labels)

    def get_image(self, index):
        assert index < len(self), f"Index {index} out of range!"
        return self.images[index]

    def is_downloaded(self):
        if not exists(join(self.root, "cifar-10-batches-py")):
            return False
        return True

    def download(self):
        if not self.is_downloaded():
            prepare_data(self.root)
        self.load_data()


class CIFAR100(Data):
    def __init__(self, root, fine=True):
        super().__init__(root=root)

        self.images = list()
        self.labels = list()
        self.fine = fine

        self.properties = pd.DataFrame({
            "Property": [
                "Name",
                "Description",
                "Number of Classes",
                "URL",
                "Date Published",
            ],
            "Value": [
                "CIFAR-100",
                "The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 "
                "million tiny images dataset. They were collected by "
                "Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.",
                100,
                "https://www.cs.toronto.edu/~kriz/cifar.html",
                2009,
            ]
        })
        self.thumbnail = join(ASSET_PATH, "images", "datasets", "CIFAR.png")

        if self.downloaded:
            self.load_data()

    def load_data(self):
        self.images = list()
        self.labels = list()
        with open(join(self.root, "cifar-100-python", "train"), 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        self.images = np.reshape(data[b"data"], (50000, 3, 32, 32))
        self.images = self.images.transpose((0, 2, 3, 1))
        self.fine_labels = data[b"fine_labels"]
        self.coarse_labels = data[b"coarse_labels"]
        self.labels = self.fine_labels if self.fine else self.coarse_labels
        self.labels = np.asarray(self.labels)

        with open(join(self.root, "cifar-100-python", "meta"), 'rb') as fo:
            classes = pickle.load(fo, encoding='bytes')
        self.fine_classes = [x.decode() for x in classes[b'fine_label_names']]
        self.coarse_classes = [x.decode() for x in
                               classes[b'coarse_label_names']]
        self.classes = self.fine_classes if self.fine else self.coarse_classes

        self.data["idx"] = range(len(self.labels))
        self.data["label"] = self.labels
        self.data["class_name"] = \
            self.data["label"].apply(lambda x: self.classes[x])
        self.data["select"] = [False] * len(self.labels)

    def get_image(self, index):
        assert index < len(self), f"Index {index} out of range!"
        img = self.images[index]
        return img

    def is_downloaded(self):
        if not exists(join(self.root, "cifar-100-python")):
            return False
        return True

    def download(self):
        if not self.is_downloaded():
            prepare_data(self.root)
        self.load_data()


def prepare_data(root):
    CIFAR10 = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    CIFAR100 = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    CIFARN = "http://www.yliuu.com/web-cifarN/files/CIFAR-N-1.zip"

    # Download CIFAR10
    name = join(root, basename(CIFAR10))
    if not exists(name):
        print("Download CIFAR10...")
        urllib.request.urlretrieve(CIFAR10, name)
    else:
        print("CIFAR10 files are existing!")

    # Extract CIFAR10
    if not exists(join(root, "cifar-10-batches-py")):
        print("Extract CIFAR10...")
        os.system(f"tar -xvf data/cifar-10-python.tar.gz -C data")
    else:
        print("CIFAR10 files are extracted!")

    # Download CIFAR100
    name = join(root, basename(CIFAR100))
    if not exists(name):
        print("Download CIFAR100...")
        urllib.request.urlretrieve(CIFAR100, name)
    else:
        print("CIFAR100 files are existing!")

    # Extract CIFAR100
    if not exists(join(root, "cifar-100-python")):
        print("Extract CIFAR100...")
        os.system(f"tar -xvf data/cifar-100-python.tar.gz -C data")
    else:
        print("CIFAR100 files are extracted!")


if __name__ == "__main__":
    prepare_data(os.path.join(__file__.split("src")[0], "data"))
