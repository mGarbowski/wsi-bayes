from trees.dataset import Dataset
from trees.tree_evaluation import evaluate_on_dataset


def main():
    dataset_1 = Dataset.load_from_file("data/data1.csv", label_col_idx=3, skip_header=True)
    dataset_2 = Dataset.load_from_file("data/data2.csv", label_col_idx=3, skip_header=True)
    evaluate_on_dataset(dataset_1, "Back pain 1", "1", "0")
    evaluate_on_dataset(dataset_2, "Back pain 2", "1", "0")


if __name__ == '__main__':
    main()
