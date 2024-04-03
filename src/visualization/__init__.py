from utils import ROOT_DIR
from visualization.visualizer import Visualizer
from visualization.tsne import TSNE
from visualization.spectral_embedding import SpectralEmbedding
from visualization.pca import PCA

visualization_algorithms = [
    TSNE(ROOT_DIR),
    SpectralEmbedding(ROOT_DIR),
    PCA(ROOT_DIR)
]
