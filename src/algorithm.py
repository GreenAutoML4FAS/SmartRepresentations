import numpy as np
import os
from os.path import dirname
from pickle import dump, load


class Algorithm:

    def __init__(self, root: str, subdir: str):
        self.data_path = \
            os.path.join(root, subdir, "data", self.__class__.__name__)
        os.makedirs(self.data_path, exist_ok=True)
        self.model_path = \
            os.path.join(root, subdir, "model", self.__class__.__name__)
        os.makedirs(self.model_path, exist_ok=True)

        self.parameter = dict()
        self.parameter_choices = dict()

        self.__name__ = self.__class__.__name__
        self.__doc__ = "No description available."  # Replace with long description
        self.__str__ = "No description available."  # Replace with short description

        self.has_results = True
        self.results = None
        self.list_results()

        self.has_model = False
        self.models = None
        self.list_models()

    ' Methods to access parameters '

    def set_parameter(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.parameter:
                self.parameter[key] = value
            else:
                raise KeyError(f"Parameter {key} not found!")

    def get_parameter(self) -> dict:
        return self.parameter

    def get_parameter_choices(self) -> dict:
        return self.parameter_choices

    ' Methods to perform inference and training '

    def inference(self, data, store_name: str, **kwargs) -> bool:
        if self.exist_result(store_name):
            return False
        else:
            res = self._inference(data, **kwargs)
            self.store_result(res, store_name)
            return True

    def _inference(self, data, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def train(self, data, store_name: str, **kwargs) -> bool:
        if self.exist_model(store_name):
            return False
        else:
            model = self._train(data, **kwargs)
            self.store_model(model, store_name)
            return True

    def _train(self, data, **kwargs):
        raise NotImplementedError

    ' Methods to access results '

    def exist_result(self, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        return os.path.exists(os.path.join(self.data_path, store_name))

    def store_result(self, result, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.data_path, store_name)
        if store_path is not None:
            os.makedirs(dirname(store_path), exist_ok=True)
            with open(store_path, "wb") as f:
                dump(result, f)
        self.results = self.list_results()

    def load_result(self, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.data_path, store_name)
        if not os.path.exists(store_path):
            return False
        if store_path is not None:
            with open(store_path, "rb") as f:
                return load(f)

    def delete_result(self, store_name: str):
        if not self.exist_result(store_name):
            return False
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.data_path, store_name)
        if store_path is not None:
            os.remove(store_path)
            self.results = self.list_results()
            return True
        return False

    def list_results(self):
        self.results = {"Name": [
            x for x in os.listdir(self.data_path) if x.endswith(".pkl")
        ]}
        return self.results

    def get_result_url(self, result):
        return os.path.join(self.data_path, result)

    ' Methods to access models '

    def exist_model(self, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        return os.path.exists(os.path.join(self.model_path, store_name))

    def store_model(self, model, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.model_path, store_name)
        if store_path is not None:
            os.makedirs(dirname(store_path), exist_ok=True)
            with open(store_path, "wb") as f:
                dump(model, f)
        self.models = self.list_models()

    def load_model(self, store_name: str):
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.model_path, store_name)
        if not os.path.exists(store_path):
            return False
        if store_path is not None:
            with open(store_path, "rb") as f:
                model = load(f)

            return self._load_model(model)

    def delete_model(self, store_name: str):
        if not self.exist_model(store_name):
            return False
        if not store_name.endswith(".pkl"):
            store_name += ".pkl"
        store_path = os.path.join(self.model_path, store_name)
        if store_path is not None:
            os.remove(store_path)
            self.models = self.list_models()
            return True
        return False

    def _load_model(self, model) -> bool:
        raise NotImplementedError

    def list_models(self):
        self.models = {"Name": [
            x for x in os.listdir(self.model_path) if x.endswith(".pkl")
        ]}
        return self.models

    def get_model_url(self, model):
        return os.path.join(self.model_path, model)
