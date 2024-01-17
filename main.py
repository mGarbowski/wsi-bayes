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
        self.parent = parents
        self.distribution = distribution

    def generate_value(self, conditions: tuple[bool, ...]) -> bool:
        true_probability = self.distribution[conditions]
        return true_probability < random()


class BayesNetwork:
    final_node: BayesNode
    nodes: list[BayesNode]

    def __init__(self, final_node: BayesNode, nodes: list[BayesNode]):
        self.final_node = final_node
        self.nodes = nodes

    def generate_data_point(self) -> tuple[bool, ...]:
        raise NotImplementedError

    def load_from_file(self):
        raise NotImplementedError

    def save_to_file(self):
        raise NotImplementedError


def main():
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


if __name__ == '__main__':
    main()
