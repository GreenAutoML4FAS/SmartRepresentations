import numpy as np
from sklearn.cluster import MeanShift as mean_shift

from clustering.cluster import ClusterAlgorithm

from utils import ASSET_PATH

citation = """
```latex
    @article{1055330,
        author={Fukunaga, K. and Hostetler, L.},
        journal={IEEE Transactions on Information Theory}, 
        title={The estimation of the gradient of a density function, with applications in pattern recognition}, 
        year={1975},
        volume={21},
        number={1},
        pages={32-40},
        keywords={},
        doi={10.1109/TIT.1975.1055330}
    }

```
"""

SHORT_DESCRIPTION = \
    f"""
---

Mean Shift is a non-parametric clustering algorithm, notable for its ability to discover the number of clusters in a dataset automatically. It works by updating candidates for centroids to be the mean of the points within a given region. Here’s a brief summary of how it functions:

---

## How to use
To cluster representations, you need to specify the following parameters according to your needs:


1. **`estimate_bandwidth` (Default: True)**: This parameter determines whether the bandwidth is automatically estimated from the input data. When set to True, the algorithm will attempt to determine the optimal bandwidth value for you. The choices are True (to estimate) or False (to use the provided bandwidth value).
2. **`bandwidth` (Default: 1.0)**: This is the bandwidth parameter of the Mean Shift algorithm. It dictates the radius of the window (or kernel) used to compute the mean. A default value of 1.0 is set, but this can be adjusted based on the scale and density of your data.
3. **`bin_seeding` (Default: False)**: When set to True, this parameter initializes the centroids more efficiently for faster convergence, typically using a discretized version of the data. The default is False, which means the algorithm will not use bin seeding. The choices are True or False.
4. **`min_bin_freq` (Default: 1)**: This parameter is used in conjunction with bin_seeding. It represents the minimum bin frequency. By default, it is set to 1, which means that bins with at least one point will be used in the seeding process.
5. **`cluster_all` (Default: 10)**: This parameter usually determines whether all points are clustered. However, the default value of 10 seems unusual for a boolean setting (typically True or False). It's possible that this represents a different or custom behavior in your implementation.
6. **`max_iter` (Default: 300)**: This sets the maximum number of iterations the algorithm runs for each convergence. With a default of 300, the algorithm will iterate up to this number of times to find clusters unless convergence is achieved sooner.

These parameters provide a comprehensive control mechanism over the clustering process, allowing users to customize the behavior of the algorithm to suit various data characteristics and clustering needs. Experimenting with different settings of these parameters can lead to more effective and efficient clustering results tailored to specific datasets and objectives.

&nbsp;

We use the method implementation from [SKLearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)

    """

DESCRIPTION = \
    f"""
---


Mean Shift is a non-parametric clustering algorithm, notable for its ability to discover the number of clusters in a dataset automatically. It works by updating candidates for centroids to be the mean of the points within a given region. Here’s a brief summary of how it functions:

1. **Initialization**: The algorithm begins by placing a circle (or a window) around each data point. The size of this window is determined by a bandwidth parameter, which dictates the radius of the circle.
2. **Shifting the Window**: For each circle, Mean Shift computes the mean of all the points within it and shifts the center of the circle to this mean. This process is iteratively repeated, effectively moving the circle (or window) towards areas of higher density.
3. **Convergence**: The shifting continues until there is no more movement in the centers, signifying convergence. Points that end up in the same window are considered to be in the same cluster.
4. **Cluster Formation**: Clusters are formed by the points that converge to the same window centers. Unlike K-Means, Mean Shift does not require specifying the number of clusters beforehand, as it can automatically find the number of clusters based on the data distribution.

Mean Shift is particularly effective for situations where the clusters are not uniform in size or shape. Its major drawback is the choice of bandwidth, which significantly influences the result. A small bandwidth can lead to over-segmentation, while a large one might merge distinct clusters. Despite this, its ability to adapt to the underlying data structure makes it a powerful tool for many clustering tasks.

---

### Paper

Title:

> **The estimation of the gradient of a density function, with applications in pattern recognition**

&nbsp;


Abstract:


> Nonparametric density gradient estimation using a generalized kernel approach is investigated. Conditions on the kernel functions are derived to guarantee asymptotic unbiasedness, consistency, and uniform consistenby of the estimates. The results are generalized to obtain a simple mean-shift estimate that can be extended in a k-nearest-neighbor approach. Applications of gradient estimation to pattern recognition are presented using clustering and intrinsic dimensionality problems, with the ultimate goal of providing further understanding of these problems in terms of density gradients!

&nbsp;

If you use Mean-Shift, please cite the following paper:

{citation}
"""


class MeanShift(ClusterAlgorithm):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "estimate_bandwidth": True,
            "bandwidth": 1.0,
            "bin_seeding": False,
            "min_bin_freq": 1,
            "cluster_all": 10,
            "max_iter": 300,
        })
        self.parameter_choices.update({
            "estimate_bandwidth": [True, False],
            "bin_seeding": [False, True],
        })
        self.__doc__ = DESCRIPTION
        self.__str__ = SHORT_DESCRIPTION

    def _inference(self, data: np.ndarray, **kwargs):
        X = data
        estimate_bandwidth = self.parameter["estimate_bandwidth"]
        bandwidth = self.parameter["bandwidth"]
        bin_seeding = self.parameter["bin_seeding"]
        min_bin_freq = self.parameter["min_bin_freq"]
        cluster_all = self.parameter["cluster_all"]
        max_iter = self.parameter["max_iter"]
        bandwidth = None if estimate_bandwidth else bandwidth
        ret = mean_shift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            min_bin_freq=min_bin_freq,
            cluster_all=cluster_all,
            max_iter=max_iter
        ).fit(X)
        labels = ret.labels_
        return labels
