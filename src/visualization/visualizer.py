import numpy as np

from algorithm import Algorithm


class Visualizer(Algorithm):

    def __init__(self, root):
        super().__init__(root=root, subdir="visualisation")

    def render(
            self,
            store_name: str,
            labels: list = None,
            colormap: list = None,
            highlight_samples: list = None,
            **kwargs
    ) -> np.ndarray:
        return self._render(
            store_name, labels, colormap, highlight_samples, **kwargs
        )

    def _render(
            self,
            store_name: str,
            labels: list = None,
            colormap: list = None,
            highlight_samples: list = None,
            **kwargs
    ) -> np.ndarray:
        raise NotImplementedError
