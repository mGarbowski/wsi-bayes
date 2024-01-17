from pprint import pprint
from random import random


class BayesNode:
    label: str
    distribution: dict[tuple[bool, ...], float]
    parents: list['BayesNode']

    def __init__(self, label: str, distribution: dict[tuple[bool, ...], float], parents: list['BayesNode'] = None):
        parents = parents if parents is not None else []
        if len(distribution) != 2 ** len(parents):
            raise ValueError("Distribution does not match the parent nodes")

        for key in distribution:
            if len(key) != len(parents):
                raise ValueError("Distribution does not match the parent nodes")

        self.label = label
        self.parents = parents
        self.distribution = distribution

    def generate_value(self, conditions: tuple[bool, ...]) -> bool:
        true_probability = self.distribution[conditions]
        return random() < true_probability

    def parent_labels(self):
        return [parent.label for parent in self.parents]


class BayesNetwork:
    """Bayes Network

    nodes are in the order they will be evaluated
    """
    final_node: BayesNode
    nodes: list[BayesNode]
    nodes_map: dict[str, int]

    def __init__(self, final_node: BayesNode, nodes: list[BayesNode]):
        self.final_node = final_node
        self.nodes = nodes
        self.nodes_map = {node.label: idx for idx, node in enumerate(nodes)}

    def get_conditions_tuple(self, values: list[bool], labels: list[str]) -> tuple[bool, ...]:
        return tuple(values[self.nodes_map[label]] for label in labels)

    def generate_data_point(self) -> tuple[bool, ...]:
        data_point = [False for _ in range(len(self.nodes))]
        for idx, node in enumerate(self.nodes):
            conditions = self.get_conditions_tuple(data_point, node.parent_labels())
            data_point[idx] = node.generate_value(conditions)

        return tuple(data_point)

    def generate_data(self, n_items: int) -> list[tuple[bool, ...]]:
        return [self.generate_data_point() for _ in range(n_items)]

    def load_from_file(self):
        raise NotImplementedError

    def save_to_file(self):
        raise NotImplementedError


def make_network():
    chair = BayesNode(
        label="Chair",
        distribution={(): 0.8}
    )

    sport = BayesNode(
        label="Sport",
        distribution={(): 0.02}
    )

    back = BayesNode(
        label="Back",
        distribution={
            (True, True): 0.9,
            (True, False): 0.2,
            (False, True): 0.9,
            (False, False): 0.01,
        },
        parents=[chair, sport]
    )

    ache = BayesNode(
        label="Ache",
        distribution={
            (True,): 0.7,
            (False,): 0.1
        },
        parents=[back]
    )

    network = BayesNetwork(ache, [chair, sport, back, ache])
    return network


def main():
    network = make_network()
    data = network.generate_data(100)

    n_chair = sum(1 for x in data if x[0])
    n_sport = sum(1 for x in data if x[1])
    n_back = sum(1 for x in data if x[2])
    n_ache = sum(1 for x in data if x[3])

    print(f"expected 80, real: {n_chair}")
    print(f"expected 2, real: {n_sport}")
    print(f"expected 18, real: {n_back}")
    print(f"expected 20, real: {n_ache}")


if __name__ == '__main__':
    main()
