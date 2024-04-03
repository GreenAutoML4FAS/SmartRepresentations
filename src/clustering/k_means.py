import numpy as np
from sklearn.cluster import KMeans as kmeans

from clustering.cluster import ClusterAlgorithm
from utils import ASSET_PATH

citation = """
```latex
    @inproceedings{macqueen1967some,
        title={Some methods for classification and analysis of multivariate observations},
        author={MacQueen, James and others},
        booktitle={Proceedings of the fifth Berkeley symposium on mathematical statistics and probability},
        volume={1},
        number={14},
        pages={281--297},
        year={1967},
        organization={Oakland, CA, USA}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

K-means is a popular unsupervised learning algorithm used for clustering data. It aims to partition a set of observations into K clusters, each represented by the mean of the points in the cluster. The process involves randomly initializing K centroids, and then iteratively refining them by assigning each data point to the nearest centroid and recalculating the centroids based on the assigned points. This process continues until the centroids stabilize, indicating that the clusters are as compact and distinct as possible. K-means is known for its simplicity and efficiency in handling large datasets, but it requires the number of clusters (K) to be specified in advance and may converge to local minima, depending on the initial centroid placement.

---

## How to use
To cluster representations, you need to specify the following parameters according to your needs:

1. **`n_clusters` (Default: 10)**: This parameter sets the number of clusters into which the data will be grouped. By default, your algorithm will segment the data into 10 distinct clusters.
2. **`init` (Default: "k-means++")**: This parameter defines the method for initializing the centroids. The default method, "k-means++", is designed to optimize the initial positioning of the centroids, potentially leading to better and more consistent clustering results. The other available option is "random", which chooses initial centroids randomly from the data points.
3. **`algorithm` (Default: "lloyd")**: This parameter determines the underlying algorithm used for the clustering process. The default is "lloyd", referring to Lloyd's algorithm, which is the standard K-Means algorithm. Other options include "auto", "full", and "elkan", each offering a different approach or optimization to the clustering process. "Auto" would typically choose the best method based on the data, "full" is a more straightforward implementation of K-Means, and "elkan" uses triangle inequality to speed up convergence.
4. **`n_init` (Default: 10)**: This parameter indicates the number of times the clustering algorithm will be run with different centroid initializations. The final results will be the best output of these runs in terms of inertia. The default value is 10, meaning the algorithm will run 10 times with different random centroids and the best solution in terms of inertia will be chosen.
5. **`max_iter` (Default: 300)**: This parameter sets the maximum number of iterations for each single run of the algorithm. The default is set to 300, which means the algorithm will perform a maximum of 300 iterations to reach convergence during each run.
6. **`random_state` (Default: 0)**: Similar to many machine learning algorithms, this parameter is used for random number generation, ensuring reproducibility of the results. Setting it to a fixed number (0 in this case) ensures that the algorithm produces the same results across different runs when the same data and parameters are used.

These parameters provide a comprehensive control mechanism over the clustering process, allowing users to customize the behavior of the algorithm to suit various data characteristics and clustering needs. Experimenting with different settings of these parameters can lead to more effective and efficient clustering results tailored to specific datasets and objectives.

&nbsp;

We use the method implementation from [SKLearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

    """

DESCRIPTION = \
    f"""
---

A detailed explanation of the K-means clustering algorithm:

1. **Overview**: K-means is an unsupervised machine learning algorithm used for clustering data into a pre-defined number of groups (K). It is widely used due to its simplicity and efficiency, especially in scenarios involving large datasets.
2. **Initialization**: The process begins by choosing 'K' initial centroids randomly. These centroids are the starting points of the clusters and are typically chosen as random data points from the dataset.
3. **Assignment Step**: Each data point in the dataset is assigned to the nearest centroid. The 'nearest' is often determined by calculating the Euclidean distance between the data point and each centroid. After this step, each centroid has a group of data points associated with it, forming K clusters.
4. **Update Step**: Once all points are assigned to clusters, the centroids are recalculated. This is typically done by taking the mean of all the points in each cluster. The centroid of each cluster now moves to this new mean location.
5. **Iteration**: Steps 3 and 4 are repeated iteratively. In each iteration, the assignments of data points to clusters might change as the centroids move. This iterative process continues until a stopping criterion is met. This could be a set number of iterations, or more commonly, the algorithm stops when the centroids no longer move significantly (indicating that the clusters are stable and well-separated).
6. **Challenges and Considerations**:
   - **Choosing K**: One of the main challenges is selecting the right number of clusters (K). There are various methods to determine the optimal K, like the Elbow Method, which involves running the algorithm with different K values and choosing the K at which the rate of decrease in the within-cluster sum of squares (WCSS) starts to diminish.
   - **Sensitivity to Initialization**: The final clusters can depend heavily on the initial random choice of centroids. To counter this, K-means is often run multiple times with different initializations, and the best clustering result (in terms of WCSS) is chosen.
   - **Local Minima**: K-means can converge to local minima, meaning it might not always find the best possible clustering solution.
   - **Spherical Clusters Assumption**: The algorithm works best when the clusters are roughly spherical and of similar size. If the clusters in the data have different shapes or densities, K-means might not perform well.
7. **Applications**: Despite these challenges, K-means is highly effective in many practical applications, including market segmentation, document clustering, image compression, and pattern recognition.

In summary, K-means is a powerful, efficient, and widely-used clustering algorithm, but its effectiveness can depend on the choice of K, the data distribution, and the initialization of centroids.

---

### Paper

Title:

> **Some methods for classification and analysis of multivariate observations**

&nbsp;


Abstract:


> Not Existing... The paper is old and abstracts seems to be not necessary!

&nbsp;

If you use K-Means, please cite the following paper:

{citation}
"""


class KMeans(ClusterAlgorithm):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "n_clusters": 10,
            "init": "k-means++",
            "algorithm": "lloyd",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 0
        })
        self.parameter_choices.update({
            "init": ["k-means++", "random"],
            "algorithm": ["lloyd", "auto", "full", "elkan"],
        })
        self.__str__ = SHORT_DESCRIPTION
        self.__doc__ = DESCRIPTION

    def _inference(self, data: np.ndarray, **kwargs):
        X = data
        n_clusters = self.parameter["n_clusters"]
        init = self.parameter["init"]
        algorithm = self.parameter["algorithm"]
        n_init = self.parameter["n_init"]
        max_iter = self.parameter["max_iter"]
        random_state = self.parameter["random_state"]
        ret = kmeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            init=init,
            algorithm=algorithm,
            max_iter=max_iter
        ).fit(X)
        labels = ret.labels_
        return labels
