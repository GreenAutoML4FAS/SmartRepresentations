import numpy as np
from sklearn.manifold import SpectralEmbedding as spectral_embedding
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from data.dataloader import Data
from visualization.visualizer import Visualizer

citation = """
```latex
    @ARTICLE{6789755,
        author={Belkin, Mikhail and Niyogi, Partha},
        journal={Neural Computation}, 
        title={Laplacian Eigenmaps for Dimensionality Reduction and Data Representation}, 
        year={2003},
        volume={15},
        number={6},
        pages={1373-1396},
        keywords={},
        doi={10.1162/089976603321780317}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

Spectral embedding is a powerful technique for reducing the dimensionality of complex datasets, transforming high-dimensional data into a lower-dimensional space while preserving the essential relationships between points. This process is based on the spectral (eigenvalue) decomposition of the graph Laplacian, constructed from the data's similarity matrix. By mapping data points using the eigenvectors of the Laplacian, spectral embedding reveals the intrinsic geometric structure of the data, facilitating effective clustering and visualization. It excels in handling non-linear and non-convex data structures, making it particularly useful for uncovering hidden patterns and structures in diverse datasets.

---

## How to use

Here's a detailed description of each parameter for spectral embedding, including their default settings and available choices for customization:

1. **`n_components` (Default: 2)**: Specifies the number of dimensions to which the data is reduced. The default setting is 2, making it suitable for 2D visualizations. Choices are [2, 3], allowing for either 2D or 3D embeddings.
2. **`affinity` (Default: "nearest_neighbors")**: Determines the method used to construct the affinity matrix. The default "nearest_neighbors" uses the k-nearest neighbors to construct the matrix. Another option is "rbf" (radial basis function), which creates the affinity matrix based on the Gaussian kernel of the distances between points.
3. **`gamma` (Default: None)**: The coefficient for the RBF kernel. When set to None, the algorithm will attempt to choose an appropriate value automatically, based on the data. This parameter is only relevant if "affinity" is set to "rbf".
4. **`random_state` (Default: None)**: A seed used by the random number generator. Setting this parameter ensures reproducible results. None means the seed is not fixed.
5. **`eigen_solver` (Default: "arpack")**: The algorithm to use for finding the eigenvectors of the affinity matrix. "arpack" is the default choice, suitable for most cases. Another option provided in the choices is "lobpcg", which can be faster for large datasets.
6. **`eigen_tol` (Default: "auto")**: The tolerance for stopping criterion for eigendecomposition when using "arpack" or "lobpcg". The default "auto" lets the algorithm choose an appropriate value based on the data.
7. **`n_neighbors` (Default: None)**: The number of neighbors to consider when constructing the affinity matrix using "nearest_neighbors". None means the algorithm will choose an optimal number based on the dataset.
8. **`visualisation_fraction` (Default: 1.0)**: Specifies the fraction of data points to use for visualization. A value of 1.0 means all data points are used. This parameter allows for downsampling in very large datasets.
9. **`figure_width` (Default: 8), `figure_height` (Default: 8)**: The dimensions of the visualization figure in inches. The default 8x8 size provides a standard square plot.

These parameters allow for significant customization of the spectral embedding process, tailoring it to specific datasets and visualization or analysis goals. Adjusting these settings can impact the quality of the embedding and the computational efficiency of the algorithm.

    """

DESCRIPTION = \
    f"""
---

Spectral embedding is a technique used in dimensionality reduction and data visualization, closely related to spectral clustering. It leverages the spectrum (eigenvalues) of the graph Laplacian, which is derived from the similarity matrix of the data, to perform a low-dimensional embedding of the dataset. The goal is to map high-dimensional data into a lower-dimensional space in such a way that the geometric relationships between points are preserved as much as possible, especially the clustering structure.

### How Spectral Embedding Works:

1. **Similarity Graph Construction**: The first step involves constructing a similarity graph from the high-dimensional data. Each node in the graph represents a data point, and edges between nodes represent the similarity between those data points. The similarity can be measured in various ways, such as the Euclidean distance or the Gaussian (radial basis function) kernel of the distance.
2. **Graph Laplacian Calculation**: Once the similarity graph is constructed, the graph Laplacian is computed. The Laplacian is a matrix that captures the difference between the degree of a node and the adjacency matrix of the graph. It reflects how well connected each vertex is to the rest of the graph.
3. **Eigenvalue Decomposition**: The next step is to perform an eigenvalue decomposition of the Laplacian matrix. The eigenvectors corresponding to the smallest eigenvalues (except for the smallest eigenvalue, which is zero) are used. These eigenvectors encode valuable information about the structure of the data.
4. **Low-dimensional Embedding**: The selected eigenvectors form a new data representation in a lower-dimensional space. By using these eigenvectors as coordinates, the original high-dimensional data points are mapped into a lower-dimensional space where the clustering structure of the data is more apparent and can be easily identified using standard clustering techniques like k-means.

### Applications and Advantages:

- **Clustering**: Spectral embedding is particularly effective for clustering non-convex clusters and discovering the intrinsic clustering structure of the data.
- **Dimensionality Reduction**: It serves as a powerful tool for reducing the dimensions of data while preserving essential relationships, making it useful for visualization and further analysis.
- **Data Visualization**: The lower-dimensional space produced by spectral embedding can be used to visualize complex high-dimensional data in 2D or 3D.

### Considerations:

- **Choice of Similarity Metric**: The choice of similarity metric and parameters (like the width of the Gaussian kernel) significantly affects the quality of the embedding and the clustering results.
- **Scalability**: While spectral embedding provides excellent results for many types of data, it can be computationally intensive for very large datasets due to the eigenvalue decomposition step.

Spectral embedding offers a unique approach to understanding and visualizing the structure of complex datasets by focusing on the relationships between data points, making it a valuable tool in the machine learning and data analysis toolkit.

---

## Paper

Title:

> **Laplacian Eigenmaps for Dimensionality Reduction and Data Representation**

&nbsp;

Abstract:

> One of the central problems in machine learning and pattern recognition is to develop appropriate representations for complex data. We consider the problem of constructing a representation for data lying on a low-dimensional manifold embedded in a high-dimensional space. Drawing on the correspondence between the graph Laplacian, the Laplace Beltrami operator on the manifold, and the connections to the heat equation, we propose a geometrically motivated algorithm for representing the high-dimensional data. The algorithm provides a computationally efficient approach to nonlinear dimensionality reduction that has locality-preserving properties and a natural connection to clustering. Some potential applications and illustrative examples are discussed.

&nbsp;

If you use Spectral Embeddings, please cite the following paper:

{citation}
"""


class SpectralEmbedding(Visualizer):
    def __init__(self, root):
        super().__init__(root=root)
        self.parameter.update({
            "n_components": 2,
            "affinity": "nearest_neighbors",
            "gamma": None,
            "random_state": None,
            "eigen_solver": "arpack",
            "eigen_tol": "auto",
            "n_neighbors": None,
            "visualisation_fraction": 1.0,
            "figure_width": 8,
            "figure_height": 8,
        })
        self.parameter_choices.update({
            "n_components": [2, 3],
            "affinity": ["nearest_neighbors", "rbf"],
            "method": ["arpack", "lobpcg"],
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
        affinity = self.parameter["affinity"]
        gamma = self.parameter["gamma"]
        random_state = self.parameter["random_state"]
        eigen_solver = self.parameter["eigen_solver"]
        eigen_tol = self.parameter["eigen_tol"]
        n_neighbors = self.parameter["n_neighbors"]

        X_embedded = spectral_embedding(
            n_components=n_components,
            affinity=affinity,
            gamma=gamma,
            random_state=random_state,
            eigen_solver=eigen_solver,
            eigen_tol=eigen_tol,
            n_neighbors=n_neighbors
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
        return img


if __name__ == "__main__":
    x = np.ndarray((100, 100))
    print(type(x).__name__)
