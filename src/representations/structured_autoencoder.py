import os
from os.path import join, exists, dirname
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# from ConfigSpace import Configuration, ConfigurationSpace
# from smac import HyperparameterOptimizationFacade, Scenario

from utils import ASSET_PATH, encode_image
from data.dataloader import Data
from representations.representation import Representation

clip_image = join(ASSET_PATH, "images", "representations", "CLIP.png")
citation = """
```latex
    @inproceedings { RudWan2019a,
          author = {Marco Rudolph and Bastian Wandt and Bodo Rosenhahn},
          title = {Structuring Autoencoders},
          booktitle = {Third International Workshop on “Robust Subspace Learning and Applications in Computer Vision” (ICCV)},
          year = {2019},
          url = {https://arxiv.org/abs/1908.02626},
          month = aug
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

Structuring AutoEncoders (SAE) are neural networks which learn a low dimensional representation of data and are additionally enriched with a desired structure in this low dimensional space. While traditional Autoencoders have proven to structure data naturally they fail to discover semantic structure that ishard to recognize in the raw data. The SAE solves the problem by enhancing a traditional Autoencoder using weak supervision to form a structured latent space. In the experiments we demonstrate, that the structured latent space allows for a much more efficient data representation for further tasks such as classification for sparsely labeled data, an efficient choice of data to label, and morphing between classes. To demonstrate the general applicability of our method, we show experiments on the benchmark image datasets MNIST, Fashion-MNIST, DeepFashion2 and on a dataset of 3D human shapes

---

## How to use




    """

DESCRIPTION = \
    f"""
---


---

## Paper

Title:

> **Structuring Autoencoders**

&nbsp;


Abstract:

> In this paper we propose Structuring AutoEncoders (SAE). SAEs are neural networks which learn a low dimensional representation of data and are additionally enriched with a desired structure in this low dimensional space. While traditional Autoencoders have proven to structure data naturally they fail to discover semantic structure that ishard to recognize in the raw data. The SAE solves the problem by enhancing a traditional Autoencoder using weak supervision to form a structured latent space. In the experiments we demonstrate, that the structured latent space allows for a much more efficient data representation for further tasks such as classification for sparsely labeled data, an efficient choice of data to label, and morphing between classes. To demonstrate the general applicability of our method, we show experiments on the benchmark image datasets MNIST, Fashion-MNIST, DeepFashion2 and on a dataset of 3D human shapes

&nbsp;

If you use the structured autoencoder, you can cite the following paper:

{citation}


"""


class AE(torch.nn.Module):
    def __init__(
            self,
            weights,
            input_size=224,
            pad=32,
            output_size=64,
    ):

        super(AE, self).__init__()
        if weights == "resnet18":
            encoder = resnet18(pretrained=True)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        self.encoder = torch.nn.Sequential(*(list(encoder.children())[:-1]))

        self.preprocessor = transforms.Compose([
            transforms.Resize(input_size + pad),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Decoder
        if output_size == 28:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 32, 7),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.Sigmoid()  # Assuming the input images are normalized to [0, 1]
            )
        elif output_size == 64:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 64, 8),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, 3,
                                   stride=2, padding=1, output_padding=1),
                nn.Sigmoid()  # Assuming the input images are normalized to [0, 1]
            )
        else:
            raise ValueError(f"Unknown output_size: {output_size}")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def preprocess(self, img):
        return self.preprocessor(img).unsqueeze(0)

    def loss(self, x, x_hat):
        # Resize images to output size
        x_hat = torch.nn.functional.interpolate(x_hat, size=x.shape[2:])
        return torch.nn.functional.mse_loss(x, x_hat)

    def encode_image(self, img):
        return self.encoder(img)


class ImageDataset(Dataset):
    def __init__(self, data: Data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.get_image(idx)  # Convert image to RGB
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        return image


class StructuredAutoencoder(Representation):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "weights": "resnet18",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "input_size": 32,
            "pad": 0,
            "output_size": 28,
            "epochs": 10,
            "lr": 1e-3,
            "batch_size": 32,
            "shuffle": True,
            "structure_key": "label",
            "structure_distance": 3,
            "structure_autoencoder": False,
            # "smac": False,
            # "smac_min_epochs": 5,
            # "smac_max_epochs": 100,
            # "smac_min_lr": 1e-5,
            # "smac_max_lr": 1e-1,
            # "smac_min_batch_size": 8,
            # "smac_max_batch_size": 64,
            # "smac_runs": 20,
        })
        self.parameter_choices.update({
            "weights": ["resnet18"],
            "device": ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            "input_size": [224, 128, 64, 32],
            "output_size": [28, 64],
            "shuffle": [True, False],
            "structure_autoencoder": [False, True],
            # "smac": [False, True],
        })
        self.__str__ = SHORT_DESCRIPTION
        self.__doc__ = DESCRIPTION
        self.ae = None

    def _train(self, data: Data, **kwargs):
        # Train the model
        epochs = self.parameter["epochs"]
        lr = self.parameter["lr"]
        batch_size = self.parameter["batch_size"]
        shuffle = self.parameter["shuffle"]
        device = self.parameter["device"]
        d = data

        def train_loop(
                epochs, lr, batch_size, shuffle, device
        ):
            ae = self._init_model()
            dataset = ImageDataset(d, ae.preprocessor)
            ae.train()

            optimizer = optim.Adam(ae.parameters(), lr=lr)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

            for epoch in range(epochs):
                print("start")
                accumulated_loss = 0
                for i, data in enumerate(dataloader):
                    imgs = data
                    imgs = imgs.to(device)
                    optimizer.zero_grad()
                    outputs = ae(imgs)
                    loss = ae.loss(outputs, imgs)
                    loss.backward()
                    optimizer.step()
                    accumulated_loss += loss.item()
                print(f'Epoch: {epoch + 1}, Loss: {accumulated_loss / len(dataloader)}')
            ae.eval()
            return ae, accumulated_loss / len(dataloader)

        # def smac_optimisation(config: Configuration, seed: int = 0)
        #     ae, loss = train_loop(
        #         epochs=config["epochs"],
        #         lr=config["lr"],
        #         batch_size=config["batch_size"],
        #         shuffle=shuffle,
        #         device=device,
        #     )
        #     return loss

        if self.parameter["smac"] and False:
            pass
            # # Define the configuration space
            # configspace = ConfigurationSpace(
            #     {"epochs": (
            #         self.parameter["smac_min_epochs"],
            #         self.parameter["smac_max_epochs"]),
            #     "lr": (
            #         self.parameter["smac_min_lr"],
            #         self.parameter["smac_max_lr"]),
            #     "batch_size": (
            #         self.parameter["smac_min_batch_size"],
            #         self.parameter["smac_max_batch_size"]),
            #     })
            #
            # # Scenario object specifying the optimization environment
            # scenario = Scenario(
            #     configspace, n_trials=self.parameter["smac_runs"])
            # # Optimize the hyperparameters
            # smac = HyperparameterOptimizationFacade(
            #     scenario, smac_optimisation,)
            # smac.optimize()
            # best_config = smac.get_best_configuration()
            # ae, _ = train_loop(
            #     epochs=best_config["epochs"],
            #     lr=best_config["lr"],
            #     batch_size=best_config["batch_size"],
            #     shuffle=shuffle,
            #     device=device,
            # )
        else:
            ae, _ = train_loop(
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                shuffle=shuffle,
                device=device,
            )

        # Return the model parameters
        params = self.parameter
        params["model_parameter"] = ae.state_dict()
        return params

    def _init_model(self):
        ae = AE(
            weights=self.parameter["weights"],
            input_size=self.parameter["input_size"],
            pad=self.parameter["pad"],
            output_size=self.parameter["output_size"],
        )
        ae.to(self.parameter["device"])
        ae.eval()
        return ae

    def _load_model(self, model) -> bool:
        if type(model) is not dict:
            return False
        model_params = model.pop("model_parameter")
        if model.keys() != self.parameter.keys():
            return False
        self.parameter.update(model)
        self.ae = self._init_model()
        self.ae.load_state_dict(model_params)
        return True

    def _inference(self, data: Data, **kwargs):
        if self.ae is None:
            raise ValueError("Model not initialized!")

        with torch.no_grad():
            features = list()
            for i in range(len(data)):
                print(f"\r({i}/{len(data)})", end="")
                img = Image.fromarray(data.get_image(i))
                image = self.ae.preprocess(img).to(
                    self.parameter["device"])
                image_features = self.ae.encode_image(image)
                features.append(image_features)
            features = torch.concatenate(features, dim=0).cpu().numpy()

        return features


if __name__ == "__main__":
    from data.cifar import CIFAR10

    ae = AE(weights="resnet18", input_size=224, pad=32, output_size=64)
    ae.to("cuda")
    ae.eval()
    img = torch.zeros((1, 3, 224, 224)).to("cuda")
    # img = ae.preprocess(img)
    encoding = ae.encoder(img)
    decoding = ae.decoder(encoding)

    # cifar10 = CIFAR10(root=join("data"))
    # model = StructuredAutoencoder(root=r"C:\Users\kaiser\Desktop\data\GreenAutoML4FAS")
    # model.inference(
    #     data=cifar10,
    #     store_path=join("data", "embeddings", "cifar10_clip.npy"),
    # )
