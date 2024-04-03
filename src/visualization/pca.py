import numpy as np
from sklearn.decomposition import PCA as pca
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import random
import os

from data.dataloader import Data
from visualization.visualizer import Visualizer

citation = """
```latex
    @article{pearson1901liii,
        title={LIII. On lines and planes of closest fit to systems of points in space},
        author={Pearson, Karl},
        journal={The London, Edinburgh, and Dublin philosophical magazine and journal of science},
        volume={2},
        number={11},
        pages={559--572},
        year={1901},
        publisher={Taylor \& Francis}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

Principal Component Analysis (PCA) is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easier to explore and visualize by reducing the number of variables. PCA achieves this by transforming the original variables into a new set of variables, the principal components, which are uncorrelated and ordered so that the first few retain most of the variation present in all of the original variables. This method is widely used for dimensionality reduction, noise reduction, data visualization, and to uncover the underlying structure of the data.

---

---

## How to use

Here's a detailed description of a parameter configuration for the PCA algorithm:

1. **`n_components` (Default: 2)**: Specifies the number of principal components to keep. By reducing the dimensionality to 2, it focuses on retaining the most significant underlying structure of the data. Choices include [2, 3], allowing for either 2D or 3D data representation.
2. **`whiten` (Default: False)**: When set to True, PCA will whiten the output, which means it will scale the components to have unit variance. This is often useful for preprocessing data before applying machine learning algorithms. Available options are [True, False].
3. **`svd_solver` (Default: "auto")**: Determines the solver to use for the decomposition. The "auto" option automatically chooses the most appropriate solver based on the type and size of the data. Other options include:
   - "full": Uses the full singular value decomposition (SVD) approach.
   - "arpack": Uses the Arnoldi decomposition to approximate the SVD.
   - "randomized": Uses a randomized algorithm for large datasets.
4. **`tol` (Default: 0.0)**: The tolerance for stopping criteria. This parameter influences the convergence of the solver. A value of 0.0 implies that the algorithm will continue until the maximum number of iterations is reached or until it achieves the maximum possible accuracy.
5. **`iterated_power` (Default: "auto")**: The number of iterations for the power method computed by the SVD solver. When set to "auto", the algorithm decides the optimal number of iterations based on the data. This parameter is relevant for the "randomized" solver and helps in obtaining a more accurate approximation of the singular value decomposition.
6. **`n_oversamples` (Default: 10)**: Specifies the number of oversamples beyond the `n_components`. This parameter is used by the "randomized" solver to ensure stability and improve the accuracy of the decomposition. It's applicable when the `svd_solver` is "randomized".
7. **`power_iteration_normalizer` (Default: "auto")**: This parameter adjusts the normalization in the power iteration method for the "randomized" solver. It provides options for the type of normalization applied during power iterations, affecting the stability and accuracy of the decomposition.
8. **`visualisation_fraction` (Default: 1.0)**: Specifies the fraction of data points to use for visualization if downsampling is desired. Set to 1.0 to use all data.
9. **`figure_width` (Default: 8), `figure_height` (Default: 8)**: Dimensions of the visualization figure in inches. Defaults to an 8x8 square.

These parameters allow you to configure the PCA algorithm to suit your specific data analysis needs, balancing between computational efficiency and the accuracy of the dimensionality reduction.

    """

DESCRIPTION = \
    f"""
---

Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (principal components) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

### How PCA Works:

1. **Standardization**: Typically, the first step in PCA is to standardize the data. Since PCA is affected by scale, you need to scale the features in your data before applying PCA, especially if the data set has measurements in different units.
2. **Covariance Matrix Computation**: PCA starts with the computation of the covariance matrix of the data, which reflects the correlation between different variables in the dataset.
3. **Eigenvalue Decomposition**: The next step involves computing the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors determine the directions of the new feature space, and the eigenvalues determine their magnitude. In other words, the eigenvalues explain the variance of the data along the new feature axes.
4. **Feature Vector Formation**: The eigenvectors are sorted by decreasing eigenvalues and chosen to form a new matrix where the eigenvectors are normalized to unit length. The eigenvectors that correspond to the largest eigenvalues (the principal components) carry the most information about the distribution of the data.
5. **Recasting the Data**: The final step is to recast the data along the principal components axes. This is achieved by multiplying the original dataset by the matrix formed from the top principal components. You can choose the number of principal components to keep based on the goal of the analysis.

### Applications of PCA:

- **Dimensionality Reduction**: PCA is most commonly used for reducing the dimensionality of data while preserving as much information as possible. This is done by discarding the components with lower information content (lower eigenvalues).
- **Visualization**: By reducing data to two or three principal components, PCA allows for the visualization of complex datasets.
- **Noise Reduction**: PCA can also be used to filter out noise from data by reconstructing the original data using only the most significant principal components.
- **Feature Extraction and Data Compression**: PCA can transform a high-dimensional dataset into a smaller set of new composite dimensions, with a minimal loss of information. This is useful in feature extraction, data compression, and speeding up machine learning algorithms.

Despite its wide range of applications, PCA is a linear technique and may not perform well when there are non-linear relationships in the data. In such cases, nonlinear dimensionality reduction techniques might be more appropriate.

---

## Paper

Title:

> **LIII. On lines and planes of closest fit to systems of points in space**

&nbsp;

Abstract:

> Not Existing... The paper is old and abstracts seems to be not necessary!

&nbsp;

If you use the PCA algorithm, please cite the following paper:

{citation}
"""


class PCA(Visualizer):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "n_components": 2,
            "whiten": False,
            "svd_solver": "auto",
            "tol": 0.0,
            "iterated_power": "auto",
            "n_oversamples": 10,
            "power_iteration_normalizer": "auto",
            "visualisation_fraction": 1.0,
            "figure_width": 8,
            "figure_height": 8,
        })
        self.parameter_choices.update({
            "n_components": [2, 3],
            "whiten": [True, False],
            "svd_solver": ["auto", "full", "arpack", "randomized"],
            "power_iteration_normalizer": ["auto", "QR", "LU", 'none'],
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
        whiten = self.parameter["whiten"]
        svd_solver = self.parameter["svd_solver"]
        tol = self.parameter["tol"]
        iterated_power = self.parameter["iterated_power"]
        n_oversamples = self.parameter["n_oversamples"]
        power_iteration_normalizer = self.parameter["power_iteration_normalizer"]

        X_embedded = pca(
            n_components=n_components,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer
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
