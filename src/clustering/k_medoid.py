import numpy as np
from sklearn_extra.cluster import KMedoids as kmedoids

from clustering.cluster import ClusterAlgorithm

from utils import ASSET_PATH

citation = """
```latex
    @article{kaufman1990partitioning,
        title={Partitioning around medoids (program pam)},
        author={Kaufman, Leonard},
        journal={Finding groups in data},
        volume={344},
        pages={68--125},
        year={1990},
        publisher={John Wiley \& Sons, Inc.}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

K-Medoids is a robust clustering algorithm used in machine learning and data analysis. It is designed to partition a dataset into a specified number of clusters, ensuring that each data point belongs to the cluster with the nearest medoid. A medoid is a representative object within a cluster, characterized by its minimal dissimilarity to all other objects in that cluster.

---

## How to use
To cluster representations, you need to specify the following parameters according to your needs:

1. **`n_clusters` (Default: 10)**: This parameter specifies the number of clusters to form. The default is set to 10, meaning the algorithm will attempt to group the data into 10 distinct clusters.
2. **`init` (Default: "k-medoids++")**: This setting determines the method used to initialize the medoids. The default "k-medoids++" is likely an advanced method for selecting initial medoids, similar in spirit to "k-means++" but adapted for K-Medoids. The other choices available are:
   - "random": Initial medoids are selected randomly.
   - "heuristic": A heuristic method is used for initial selection, which might involve some form of pre-analysis of the data.
   - "k-medoids++": An optimized method for initial medoid selection, likely designed to spread out the initial medoids.
   - "build": A specific, possibly more systematic approach to building the initial set of medoids.
3. **`method` (Default: "alternate")**: This parameter defines the algorithmic approach to be used in the clustering process. The default is "alternate". Available choices are:
   - "alternate": This might refer to an alternating approach to optimizing medoids and assigning points to clusters.
   - "pam": Refers to Partitioning Around Medoids, a classic K-Medoids clustering method.
4. **`max_iter` (Default: 300)**: This parameter sets the maximum number of iterations the algorithm will run to converge on a solution. It's set to 300 by default, which is the upper limit on the number of iterations the algorithm will perform.
5. **`random_state` (Default: 0)**: This is used for seeding the random number generator, ensuring reproducibility of the clustering results. A value of 0 means that the random number generator can be initialized in a way that the output of the algorithm is consistent across runs with the same data and parameters.

These parameters provide a comprehensive control mechanism over the clustering process, allowing users to customize the behavior of the algorithm to suit various data characteristics and clustering needs. Experimenting with different settings of these parameters can lead to more effective and efficient clustering results tailored to specific datasets and objectives.

&nbsp;

We use the method implementation from [SKLearn](https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html)

    """

DESCRIPTION = \
    f"""
---

A detailed introduction into the K-Medoid clustering algorithm:

1. **Initialization**: The algorithm starts by selecting a predefined number of data points from the dataset as the initial medoids. This selection can be random or based on certain criteria.
2. **Assignment**: Each data point in the dataset is then assigned to the closest medoid. The closeness is usually determined by a distance measure, such as Euclidean, Manhattan, or any other appropriate metric for the data.
3. **Update**: After the assignment, the algorithm checks if replacing one of the medoids with a non-medoid point reduces the total dissimilarity within the cluster. If a swap leads to a more optimal configuration (i.e., lower total dissimilarity), the medoid is updated.
4. **Iteration**: The assignment and update steps are repeated iteratively. In each iteration, the algorithm reassesses and adjusts the composition of clusters and their respective medoids.
5. **Convergence**: The process continues until the medoids no longer change, indicating that each data point is grouped with its nearest medoid and the clusters are as compact as possible.

K-Medoids is favored for its robustness, particularly in datasets with noise and outliers, as the algorithm chooses actual data points as cluster centers (medoids) rather than computing the mean. It's suitable for a variety of applications, especially where interpretability is important, as the medoids are representative of the members of their clusters.

---

### Paper

Title:

> **Partitioning around medoids (Program PAM)**

&nbsp;


Abstract:


> Not Existing... The paper is old and abstracts seems to be not necessary!

&nbsp;

If you use K-Medoid, please cite the following paper:

{citation}
"""


class KMedoids(ClusterAlgorithm):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "n_clusters": 10,
            "init": "k-medoids++",
            "method": "alternate",
            "max_iter": 300,
            "random_state": 0
        })
        self.parameter_choices.update({
            "init": ["random", "heuristic", "k-medoids++", "build"],
            "method": ["alternate", "pam"],
        })
        self.__doc__ = DESCRIPTION
        self.__str__ = SHORT_DESCRIPTION

    def _inference(self, data: np.ndarray, **kwargs):
        X = data
        n_clusters = self.parameter["n_clusters"]
        init = self.parameter["init"]
        method = self.parameter["method"]
        max_iter = self.parameter["max_iter"]
        random_state = self.parameter["random_state"]
        ret = kmedoids(
            n_clusters=n_clusters,
            random_state=random_state,
            init=init,
            method=method,
            max_iter=max_iter
        ).fit(X)
        labels = ret.labels_
        return labels
