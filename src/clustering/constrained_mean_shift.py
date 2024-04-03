import numpy as np

from clustering.cluster import ClusterAlgorithm
from CMS import CMS

from utils import ASSET_PATH

citation = """
```latex
    @inproceedings{schier2022constrained,
        title={Constrained Mean Shift Clustering},
        author={Schier, Maximilian and Reinders, Christoph and Rosenhahn, Bodo},
        booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
        year={2022},
        organization={SIAM}
    }

```
"""

SHORT_DESCRIPTION = \
    f"""
---

This is a constrained version of the Mean Shift clustering algorithm.
Mean Shift is a non-parametric clustering algorithm, notable for its ability to discover the number of clusters in a dataset automatically. It works by updating candidates for centroids to be the mean of the points within a given region. Here’s a brief summary of how it functions:
The constraints added to this algorithm are cannot-link constraints. These constraints provide a guidance in constrained clustering indicating that the respective pair should not be assigned to the same cluster. The algorithm introduces a density-based integration of the constraints to generate individual distributions of the sampling points per cluster. It also alleviates the (in general very sensitive) mean shift bandwidth parameter by proposing an adaptive bandwidth adjustment which is especially useful for clustering imbalanced data sets.
---

## How to use
To cluster representations, you need to specify the following parameters according to your needs:

1. **`bandwidth`**: This parameter sets the bandwidth used in the algorithm. 
2. **`max_iterations`**: This specifies the maximum number of iterations the algorithm will perform. It's a cap on the number of times the algorithm will update the cluster centers in an attempt to converge.
3. **`blurring`**: A boolean parameter that dictates the type of mean shift algorithm used. If set to True, the algorithm will use blurring mean shift. Otherwise, it will use the non-blurring mean shift. The blurring version updates all points at each iteration, whereas the non-blurring version updates cluster centers only.
4. **`kernel`**: This determines the type of kernel used in the algorithm. If set to 'ball', a ball kernel is used. If it's a scalar float in the range [0, 1), a truncated Gaussian kernel is used, with the scalar value representing the truncation boundary.
5. **`use_cuda`**: A boolean parameter indicating whether to use CUDA for computations. This requires a CUDA-Toolkit to be installed and enables GPU acceleration, potentially speeding up computations.
6. **`c_scale`**: This is the constraint scaling parameter that determines the influence of constraints in the clustering process. It's a critical parameter in adjusting how constraints affect the formation of clusters.
7. **`label_merge_k`**: Used in the connectivity matrix for label extraction using connected components. It configures the minimum closeness in terms of the kernel for two modes to be considered as connected. This parameter helps in defining how clusters are merged based on closeness.
8. **`label_merge_b`**: Another parameter for the connectivity matrix affecting label extraction. It sets the threshold for the worst constraint reduction below which two modes are never considered connected. Setting this to 0 effectively disables this feature.
9. **`stop_early`**: If set to True, the algorithm stops before reaching the maximum number of iterations if cluster centers become stationary. This is particularly effective when using a fixed bandwidth, as it can save computational time by terminating the process once clusters are stable.


Each of these parameters plays a specific role in tailoring the Constrained Mean Shift Clustering algorithm to the data and the specific requirements of the clustering task.
These parameters provide a comprehensive control mechanism over the clustering process, allowing users to customize the behavior of the algorithm to suit various data characteristics and clustering needs. Experimenting with different settings of these parameters can lead to more effective and efficient clustering results tailored to specific datasets and objectives.

&nbsp;

We use the method implementation from [https://github.com/m-schier/cms](https://github.com/m-schier/cms)

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

> **Constrained Mean Shift Clustering**

&nbsp;


Abstract:


> In this paper, we present Constrained Mean Shift (CMS), a novel approach for mean shift clustering under sparse supervision using cannot-link constraints. The constraints provide a guidance in constrained clustering indicating that the respective pair should not be assigned to the same cluster. Our method introduces a density-based integration of the constraints to generate individual distributions of the sampling points per cluster. We also alleviate the (in general very sensitive) mean shift bandwidth parameter by proposing an adaptive bandwidth adjustment which is especially useful for clustering imbalanced data sets. Several experiments show that our approach achieves better performance compared to state-of-the-art methods both clustering synthetic data sets as well as clustering encoded features of real-world image data sets.

&nbsp;

If you use Constrained Mean-Shift, please cite the following paper:

{citation}

Code is available at:

> [https://github.com/m-schier/cms](https://github.com/m-schier/cms)

"""


class ConstrainedMeanShift(ClusterAlgorithm):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "bandwidth": 1.0,
            "max_iterations": 1000,
            "blurring": True,
            "kernel": .02,
            "use_cuda": False,
            "c_scale": .5,
            "label_merge_k": .95,
            "label_merge_b": .1,
            "stop_early": True,
            "constraints": "",
        })

        self.parameter_choices.update({
            "blurring": [False, True],
            "use_cuda": [False, True],
            "stop_early": [False, True],
        })
        self.__doc__ = DESCRIPTION
        self.__str__ = SHORT_DESCRIPTION

    def _inference(self, data: np.ndarray, **kwargs):
        X = data
        bandwidth = self.parameter["bandwidth"]
        blurring = self.parameter["blurring"]
        kernel = self.parameter["kernel"]
        use_cuda = self.parameter["use_cuda"]
        c_scale = self.parameter["c_scale"]
        max_iterations = self.parameter["max_iterations"]
        label_merge_k = self.parameter["label_merge_k"]
        label_merge_b = self.parameter["label_merge_b"]
        stop_early = self.parameter["stop_early"]
        constraints = self.parameter["constraints"]
        # Transform constraints of the string form a,b;c,d;e,f; to [[a,b],[c,d],[e,f]]
        constraints = np.array([list(map(int, c.split(","))) for c in constraints.split(";") if c != ""])
        if constraints.shape[0] == 0:
            constraints = np.zeros((0, 2), dtype=np.int32)
        ret = CMS(
            h=bandwidth,
            blurring=blurring,
            kernel=kernel,
            c_scale=c_scale,
            use_cuda=use_cuda,
            max_iterations=max_iterations,
            label_merge_k=label_merge_k,
            label_merge_b=label_merge_b,
            stop_early=stop_early,
            verbose=False,
            save_history=False,
        ).fit(X, constraints)

        labels = ret.labels_
        return labels
