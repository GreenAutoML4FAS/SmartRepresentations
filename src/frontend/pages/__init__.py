import frontend.pages.root as root
import frontend.pages.dashboard as dashboard
import frontend.pages.datasets as datasets
import frontend.pages.representations as representations
import frontend.pages.clustering as clustering
import frontend.pages.visualization as visualization
import frontend.pages.current_configuration as current_config

from frontend.pages.root import *
from frontend.pages.dashboard import *
from frontend.pages.datasets import *
from frontend.pages.representations import *
from frontend.pages.clustering import *
from frontend.pages.visualization import *
from frontend.pages.current_configuration import *

pages = {
    "/": root.page,
    "Dashboard": dashboard.page,
    "Datasets": datasets.page,
    "Representations": representations.page,
    "Clustering": clustering.page,
    "Visualization": visualization.page,
}

__all__ = [
    "pages",
    *root.__all__,
    *dashboard.__all__,
    *datasets.__all__,
    *representations.__all__,
    *clustering.__all__,
    *visualization.__all__,
    *current_config.__all__,
]
