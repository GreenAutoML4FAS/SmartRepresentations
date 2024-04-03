from utils import ROOT_DIR
from clustering.cluster import ClusterAlgorithm
from clustering.k_means import KMeans
from clustering.k_medoid import KMedoids
from clustering.mean_shift import MeanShift
from clustering.constrained_mean_shift import ConstrainedMeanShift

cluster_algorithms = [
    KMeans(ROOT_DIR),
    KMedoids(ROOT_DIR),
    MeanShift(ROOT_DIR),
    ConstrainedMeanShift(ROOT_DIR)
]
