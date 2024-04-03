from utils import ROOT_DIR
from representations.representation import Representation
from representations.clip_embedding import CLIP
from representations.structured_autoencoder import StructuredAutoencoder
from representations.resnet import ResNet

representations = [
    CLIP(ROOT_DIR),
    ResNet(ROOT_DIR),
    StructuredAutoencoder(ROOT_DIR),
]
