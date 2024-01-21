import argparse

from bayes.bayes_network import BayesNetwork


def generate_csv_formatted_data(item_count: int, network_filename: str):
    network = BayesNetwork.load_from_file(network_filename)
    data = network.generate_data(item_count)
    csv_data = [
        ",".join(network.labels()),
        *[",".join(map(lambda x: "1" if x else "0", item)) for item in data]
    ]
    csv_string = "\n".join(csv_data)
    print(csv_string)


def main():
    parser = argparse.ArgumentParser(description="Generate data from Bayes network in .csv format.")
    parser.add_argument("items_count", type=int, help="The number of items to generate.")
    parser.add_argument("--file", "-f", type=str, help="The file representing Bayes network.")

    args = parser.parse_args()
    generate_csv_formatted_data(args.items_count, args.file)


if __name__ == "__main__":
    main()
