import pandas as pd
import os


class Data:
    def __init__(
            self,
            root,
    ):
        self.root = os.path.join(root, "data")

        self.data = pd.DataFrame()
        self.properties = pd.DataFrame()
        self.downloaded = self.is_downloaded()

        self.__name__ = self.__class__.__name__
        self.thumbnail = None

    def is_downloaded(self):
        return False

    def download(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data)

    def get_image(self, index):
        raise NotImplementedError

    def get_label(self, index):
        assert index < len(self), f"Index {index} out of range!"
        assert "label" in self.data.columns, "No label column in data!"
        return self.data["label"].iloc[index]
