import json
from random import random

DataPoint = tuple[bool, ...]
Conditions = tuple[bool, ...]


class BayesNode:
    label: str
    distribution: dict[Conditions, float]
    parents: list['BayesNode']

    def __init__(self, label: str, distribution: dict[Conditions, float], parents: list['BayesNode'] = None):
        parents = parents if parents is not None else []
        if len(distribution) != 2 ** len(parents):
            raise ValueError("Distribution does not match the parent nodes")

        for key in distribution:
            if len(key) != len(parents):
                raise ValueError("Distribution does not match the parent nodes")

        self.label = label
        self.parents = parents
        self.distribution = distribution

    def generate_value(self, conditions: Conditions) -> bool:
        true_probability = self.distribution[conditions]
        return random() < true_probability

    def parent_labels(self) -> list[str]:
        return [parent.label for parent in self.parents]

    def as_dict(self) -> dict:
        distribution = [
            {
                "conditions": list(conditions),
                "probability": probability
            }
            for conditions, probability in self.distribution.items()
        ]

        return {
            "label": self.label,
            "distribution": distribution,
            "parent_labels": self.parent_labels()
        }


class BayesNetwork:
    """Bayes Network

    nodes are in the order they will be evaluated and in the order they will appear as columns
    """
    nodes: list[BayesNode]
    nodes_map: dict[str, int]

    def __init__(self, nodes: list[BayesNode]):
        self.nodes = nodes
        self.nodes_map = {node.label: idx for idx, node in enumerate(nodes)}

    def get_conditions_tuple(self, values: list[bool], labels: list[str]) -> Conditions:
        return tuple(values[self.nodes_map[label]] for label in labels)

    def generate_data_point(self) -> DataPoint:
        data_point = [False for _ in range(len(self.nodes))]
        for idx, node in enumerate(self.nodes):
            conditions = self.get_conditions_tuple(data_point, node.parent_labels())
            data_point[idx] = node.generate_value(conditions)

        return tuple(data_point)

    def generate_data(self, n_items: int) -> list[DataPoint]:
        return [self.generate_data_point() for _ in range(n_items)]

    @classmethod
    def load_from_file(cls, filename: str) -> 'BayesNetwork':
        with open(filename, mode="rt", encoding="utf-8") as file:
            node_dicts = json.load(file)

        node_map: dict[str, BayesNode] = dict()
        nodes = []
        for nd in node_dicts:
            distribution = {
                tuple(d_row["conditions"]): d_row["probability"]
                for d_row in nd["distribution"]
            }
            parents = [node_map[label] for label in nd["parent_labels"]]
            node = BayesNode(nd["label"], distribution, parents)
            node_map[node.label] = node
            nodes.append(node)

        return cls(nodes)

    def save_to_file(self, filename: str):
        with open(filename, mode="wt", encoding="utf-8") as file:
            json.dump([node.as_dict() for node in self.nodes], file, indent=2)
