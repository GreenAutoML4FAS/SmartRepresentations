import re
from taipy import Config
from taipy.gui import notify, gui
import taipy as tp
import numpy as np
import datetime as dt
import time

from data import Data
from representations import Representation
from clustering import ClusterAlgorithm
from visualization import Visualizer

''' Methods performed in representations page'''


def train_representation(
        data: Data, method: Representation, model_name: str
):
    ret = method.train(data, model_name)
    return ret


def infer_representation(
        data: Data, method: Representation, name: str
):
    ret = method.inference(data, name)
    return ret


''' Methods performed in clustering page'''


def infer_clustering(
        dataset: Data,
        representation: Representation,
        embedding_name: str,
        method: ClusterAlgorithm,
        name: str,
):
    # Load embedding
    embedding = representation.load_result(embedding_name)
    ret = method.inference(embedding, name)
    return ret


def train_clustering(
        dataset: Data,
        representation: Representation,
        embedding_name: str,
        method: ClusterAlgorithm,
        name: str,
):
    # Load embedding
    raise NotImplementedError


''' Methods performed in visualization page'''


def visualize(
        dataset: Data,
        representation: Representation,
        embedding_name: str,
        method: Visualizer,
        name: str,
):
    embedding = representation.load_result(embedding_name)
    ret = method.inference(
        data=dataset,
        store_name=name,
        embedding=embedding
    )
    return ret


''' Configurations for the frontend '''


class NodeConfigs:
    def __init__(self):
        self.dataset = Config.configure_data_node("dataset")
        self.representation = Config.configure_data_node("representation")
        self.clustering = Config.configure_data_node("clustering")
        self.visualisation = Config.configure_data_node("visualisation")
        self.representation_result_name = \
            Config.configure_data_node("representation_result_name")
        self.clustering_result_name = \
            Config.configure_data_node("clustering_result_name")
        self.visualisation_result_name = \
            Config.configure_data_node("visualisation_result_name")
        self.finished = Config.configure_data_node("finished")


class TaskConfigs:
    def __init__(self, node_cfg: NodeConfigs):
        self.train_representation = Config.configure_task(
            "representations_train", train_representation,
            [node_cfg.dataset, node_cfg.representation,
             node_cfg.representation_result_name],
            node_cfg.finished
        )
        self.inference_representation = Config.configure_task(
            "representations_inference", infer_representation,
            [node_cfg.dataset, node_cfg.representation,
             node_cfg.representation_result_name],
            node_cfg.finished
        )
        self.train_clustering = Config.configure_task(
            "clustering_train", train_clustering,
            [node_cfg.dataset, node_cfg.representation,
             node_cfg.representation_result_name,
             node_cfg.clustering, node_cfg.clustering_result_name],
            node_cfg.finished
        )
        self.inference_clustering = Config.configure_task(
            "clustering_inference", infer_clustering,
            [node_cfg.dataset, node_cfg.representation,
             node_cfg.representation_result_name,
             node_cfg.clustering, node_cfg.clustering_result_name],
            node_cfg.finished
        )
        self.visualize = Config.configure_task(
            "visualize", visualize,
            [node_cfg.dataset, node_cfg.representation,
             node_cfg.representation_result_name,
             node_cfg.visualisation, node_cfg.visualisation_result_name],
            node_cfg.finished
        )


class ScenarioConfigs:
    def __init__(self, task_cfg: TaskConfigs):
        self.train_representation = Config.configure_scenario(
            id="TrainRepresentation",
            task_configs=[task_cfg.train_representation]
        )
        self.infer_representation = Config.configure_scenario(
            id="InferRepresentation",
            task_configs=[task_cfg.inference_representation]
        )
        self.train_clustering = Config.configure_scenario(
            id="TrainClustering",
            task_configs=[task_cfg.train_clustering]
        )
        self.infer_clustering = Config.configure_scenario(
            id="InferClustering",
            task_configs=[task_cfg.inference_clustering]
        )
        self.visualize = Config.configure_scenario(
            id="Visualize",
            task_configs=[task_cfg.visualize]
        )


node_configs = NodeConfigs()
task_configs = TaskConfigs(node_configs)
scenario_configs = ScenarioConfigs(task_configs)

__all__ = ["node_configs", "task_configs", "scenario_configs"]
