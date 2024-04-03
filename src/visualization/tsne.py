import numpy as np
from sklearn.manifold import TSNE as tsne
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from data.dataloader import Data
from visualization.visualizer import Visualizer

citation = """
```latex
    @article{van2008visualizing,
        title={Visualizing data using t-SNE.},
        author={Van der Maaten, Laurens and Hinton, Geoffrey},
        journal={Journal of machine learning research},
        volume={9},
        number={11},
        year={2008}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a sophisticated technique for reducing high-dimensional data into a lower-dimensional space, making it ideal for visualization purposes. Essentially, t-SNE excels at revealing patterns, clusters, and relationships between data points by mapping similar items closer together and dissimilar items farther apart in a visually interpretable form, typically in two or three dimensions. It achieves this by converting high-dimensional distances into probabilities that reflect similarities, then optimizing these in the lower-dimensional space to preserve local structures. While highly effective for exploring and visualizing complex datasets, t-SNE's performance is dependent on parameter settings, such as perplexity, and can be computationally demanding. Despite this, its ability to uncover hidden structures in data makes it a valuable tool for data analysis across various fields.

---

## How to use



Here's a detailed description of a parameter configuration for a t-SNE algorithm:


1. **`n_components` (Default: 2)**: The dimension of the embedded space. By default, data will be reduced to 2 dimensions. Choices are [2, 3] for visualizing in 2D or 3D.
2. **`perplexity` (Default: 30.0)**: A measure of the effective number of neighbors. The perplexity value balances attention between local and global aspects of your data, with a default setting of 30.0.
3. **`early_exaggeration` (Default: 12.0)**: Controls how much the algorithm exaggerates the distances between high-dimensional points in the initial optimization stages. A higher value helps separate clusters. The default is 12.0.
4. **`n_iter` (Default: 1000)**: The maximum number of iterations for the optimization. Set to 1000 by default to allow the algorithm sufficient time to converge.
5. **`n_iter_without_progress` (Default: 300)**: Stops optimization if no progress is made in terms of minimizing the Kullback-Leibler divergence in this many iterations. Default is 300 iterations.
6. **`min_grad_norm` (Default: 1e-07)**: The optimization will stop when the gradient norm is below this threshold, indicating a minimum has been reached.
7. **`init` (Default: "pca")**: Initialization method for embedding. "pca" starts the embedding on the first principal components, while "random" initializes randomly. Choices are ["random", "pca"].
8. **`random_state` (Default: 0)**: A seed used by the random number generator for reproducibility of results.
9. **`method` (Default: "barnes_hut")**: The algorithm used for gradient descent. "barnes_hut" offers a faster approximation suitable for larger datasets. Choices are ["barnes_hut", "exact"] for exact gradient computation.
10. **`angle` (Default: 0.5)**: A trade-off between speed and accuracy for the "barnes_hut" method. Lower values increase accuracy but also runtime.
11. **`visualisation_fraction` (Default: 1.0)**: Specifies the fraction of data points to use for visualization if downsampling is desired. Set to 1.0 to use all data.
12. **`figure_width` (Default: 8), `figure_height` (Default: 8)**: Dimensions of the visualization figure in inches. Defaults to an 8x8 square.

This parameter configuration provides a comprehensive setup for customizing and running the t-SNE algorithm, allowing for flexibility in balancing between computational efficiency and the quality of the visualization.
By adjusting these parameters, you can tailor the model to your specific requirements,
ensuring optimal performance during inference. Remember, no additional training is required; 
simply configure these settings, safe the model and the model is ready to deliver 
its full potential.

    """

DESCRIPTION = \
    f"""
---

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a powerful machine learning algorithm for dimensionality reduction, widely used for visualizing high-dimensional data in a low-dimensional space, typically two or three dimensions. Developed by Laurens van der Maaten and Geoffrey Hinton, t-SNE excels in capturing the local structure of high-dimensional data and grouping similar data points together in the low-dimensional space.

&nbsp;

The core idea behind t-SNE is to convert the distances between data points in high-dimensional space into conditional probabilities that represent similarities. These probabilities are then used to construct a similar probability distribution in the low-dimensional space, with the aim of minimizing the difference between the two distributions. This is achieved through a gradient descent optimization process.

&nbsp;

t-SNE starts by calculating the probability of similarity of points in the high-dimensional space, with a focus on preserving small pairwise distances or local similarities. It then maps these points to a lower-dimensional space in such a way that similar objects are modeled by nearby points, and dissimilar objects are modeled by distant points with high probability.

&nbsp;

One of the key features of t-SNE is its use of a Student’s t-distribution (hence the "t" in t-SNE) in the low-dimensional space. This helps to alleviate the crowding problem, where points tend to cluster together too tightly, by effectively spreading out the points that are moderately far apart in the high-dimensional space.

&nbsp;

While t-SNE is particularly useful for visualizing clusters or groups within data, it's computationally intensive and sensitive to the choice of hyperparameters, such as the perplexity, which influences the balance between local and global aspects of your data. Despite these considerations, t-SNE remains a popular tool for exploratory data analysis, especially in fields like bioinformatics, finance, and text analysis, where understanding the structure and relationships within complex datasets is crucial.

---

## Paper

Title:

> **Visualizing data using t-SNE.**

&nbsp;

Abstract:

>We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large data sets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of data sets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the data sets.

&nbsp;

If you use T-SNE, please cite the following paper:

{citation}
"""


class TSNE(Visualizer):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "n_components": 2,
            "perplexity": 30.0,
            "early_exaggeration": 12.0,
            "n_iter": 1000,
            "n_iter_without_progress": 300,
            "min_grad_norm": 1e-07,
            "init": "pca",
            "random_state": 0,
            "method": "barnes_hut",
            "angle": 0.5,
            "visualisation_fraction": 1.0,
            "figure_width": 8,
            "figure_height": 8,
        })
        self.parameter_choices.update({
            "n_components": [2, 3],
            "init": ["random", "pca"],
            "method": ["barnes_hut", "exact"],
        })

        self.__str__ = SHORT_DESCRIPTION
        self.__doc__ = DESCRIPTION

    def _inference(
            self,
            data: Data,
            embedding: np.ndarray,
            **kwargs
    ):
        n_components = self.parameter["n_components"]
        perplexity = self.parameter["perplexity"]
        early_exaggeration = self.parameter["early_exaggeration"]
        n_iter = self.parameter["n_iter"]
        n_iter_without_progress = self.parameter["n_iter_without_progress"]
        min_grad_norm = self.parameter["min_grad_norm"]
        init = self.parameter["init"]
        random_state = self.parameter["random_state"]
        method = self.parameter["method"]
        angle = self.parameter["angle"]

        X_embedded = tsne(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            n_iter=n_iter,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            init=init,
            random_state=random_state,
            method=method,
            angle=angle
        ).fit_transform(embedding)
        output = dict()
        output["coordinates"] = X_embedded
        return output

    def _render(
            self,
            store_name: str,
            labels: list = None,
            colormap: list = None,
            highlight_samples: list = None,
            **kwargs
    ) -> np.ndarray:

        ret = self.load_result(store_name)
        coordinates = ret["coordinates"]
        if labels is None:
            labels = [0] * coordinates.shape[0]
        is_3d = coordinates.shape[1] == 3

        unique_labels = np.unique(labels).astype(int)
        cmap = plt.cm.get_cmap('hsv', len(unique_labels))  # Replace 'hsv' with any other colormap

        inds = np.arange(coordinates.shape[0])
        visualisation_fraction = self.parameter["visualisation_fraction"]
        if 0.0 < visualisation_fraction < 1.0:
            num_samples = int(coordinates.shape[0] * visualisation_fraction)
            inds = np.random.choice(coordinates.shape[0], num_samples,
                                    replace=False)

        df = pd.DataFrame()
        df['label'] = np.asarray(labels, dtype=int)[inds]
        df['tsne-1d'] = coordinates[inds, 0]
        df['tsne-2d'] = coordinates[inds, 1]
        if is_3d:
            df['tsne-3d'] = coordinates[inds, 2]

        fig = plt.figure(figsize=(
            self.parameter["figure_width"], self.parameter["figure_height"])
        )
        if not is_3d:
            ax = plt.gca()
            scatter = ax.scatter(
                x=df["tsne-1d"], y=df["tsne-2d"],
                c=df["label"], alpha=0.6, s=8, cmap=cmap, marker="."
            )
        else:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                xs=df["tsne-1d"], ys=df["tsne-2d"], zs=df["tsne-3d"],
                c=df["label"], alpha=0.6, s=8, cmap=cmap, marker="."
            )

        if highlight_samples is not None:
            df2 = pd.DataFrame()
            df2['label'] = np.asarray(labels)[highlight_samples]
            df2['tsne-1d'] = coordinates[highlight_samples, 0]
            df2['tsne-2d'] = coordinates[highlight_samples, 1]
            if is_3d:
                df2['tsne-3d'] = coordinates[highlight_samples, 2]
            if not is_3d:
                scatter = ax.scatter(
                    x=df2["tsne-1d"], y=df2["tsne-2d"],
                    c=df2["label"], alpha=0.6, s=50, cmap=cmap, marker="D",
                )
            else:
                scatter = ax.scatter(
                    xs=df2["tsne-1d"], ys=df2["tsne-2d"], zs=df2["tsne-3d"],
                    c=df2["label"], alpha=1.0, s=50, cmap=cmap, marker="D",
                )

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'{l}',
                   markerfacecolor=cmap(l), markersize=10) for l in unique_labels]

        # Set title
        ax.set_title("TSNE")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if is_3d:
            ax.axes.get_zaxis().set_visible(False)

        ax.legend(handles=legend_elements, loc='upper right', title="Labels")

        plt.show()

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        map = (img[:, :, 1] > 250) & (img[:, :, 0] > 250) & (img[:, :, 2] > 250)
        img[:, :, 3] = 255 - map * 255
        return img
