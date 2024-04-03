import os
import numpy as np
import pandas as pd
from os.path import join, exists
from torchvision.datasets import GTSRB as gtsrb

from data.dataloader import Data
from utils import ASSET_PATH

classes = {
    0: '20 km/h',
    1: '30 km/h',
    2: '50 km/h',
    3: '60 km/h',
    4: '70 km/h',
    5: '80 km/h',
    6: '80 km/h end',
    7: '100 km/h',
    8: '120 km/h',
    9: 'No overtaking',
    10: 'No overtaking by lorries',
    11: 'Crossroad with secondary way',
    12: 'Main road',
    13: 'Give way',
    14: 'Stop',
    15: 'Road up',
    16: 'Road ahead',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows',
    25: 'Construction ahead',
    26: 'Traffic signals',
    27: 'Pedestrian crossing',
    28: 'School crossing',
    29: 'Cycle path',
    30: 'Snow',
    31: 'Animals',
    32: 'Restrictions ends',
    33: 'Go right',
    34: 'Go left',
    35: 'Go straight',
    36: 'Go right or straight',
    37: 'Go left or straight',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout',
    41: 'End of no-overtaking zone',
    42: 'End of no-overtaking zone by lorries'
}


class GTSRB(Data):

    def __init__(self, root):
        super().__init__(root=root)
        self.gtsrb = None
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
        self.thumbnail = join(ASSET_PATH, "images", "datasets", "GTSRB.jpg")

        if self.downloaded:
            self.load_data()

    def load_data(self):
        self.gtsrb = gtsrb(root=self.root, download=True)
        self.labels = np.asarray([x[1] for x in self.gtsrb._samples])

        self.data["idx"] = range(len(self.labels))
        self.data["label"] = self.labels
        self.data["class_name"] = \
            self.data["label"].apply(lambda x: classes[x])
        self.data["select"] = [False] * len(self.labels)

    def get_image(self, index):
        assert index < len(self), f"Index {index} out of range!"
        img, label = self.gtsrb.__getitem__(index)
        img = np.array(img)
        return img

    def is_downloaded(self):
        if not exists(join(self.root, "gtsrb")):
            return False
        return True

    def download(self):
        if not self.is_downloaded():
            prepare_data(self.root)
        self.load_data()


def prepare_data(root):
    gtsrb(root=root, download=True)
