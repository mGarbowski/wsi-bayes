from bayes_network import BayesNode, BayesNetwork


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

    network = BayesNetwork([chair, sport, back, ache])
    return network


def check_expected_values(network: BayesNetwork, n_runs = 25):
    n_chair = 0
    n_sport = 0
    n_back = 0
    n_ache = 0

    for _ in range(n_runs):
        data = network.generate_data(100)
        n_chair += sum(1 for x in data if x[0])
        n_sport += sum(1 for x in data if x[1])
        n_back += sum(1 for x in data if x[2])
        n_ache += sum(1 for x in data if x[3])

    n_chair /= n_runs
    n_sport /= n_runs
    n_back /= n_runs
    n_ache /= n_runs

    print(f"expected 80, real: {n_chair}")
    print(f"expected 2, real: {n_sport}")
    print(f"expected 18, real: {n_back}")
    print(f"expected 20, real: {n_ache}")


def main():
    network = make_network()
    check_expected_values(network)
    network.save_to_file("./docs/network.json")
    network_2 = BayesNetwork.load_from_file("../docs/network.json")
    check_expected_values(network_2)


if __name__ == '__main__':
    main()
