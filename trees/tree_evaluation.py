from trees.dataset import Dataset
from trees.decision_trees import DecisionTreeClassifier


def avg(values: list) -> float:
    return sum(values) / len(values)


def evaluate_on_dataset(
        dataset: Dataset,
        name: str,
        positive_label: str,
        negative_label: str,
        split_ratio: float = 0.6,
        n_times: int = 25
):
    evaluations = []
    n_test_set_samples = 0
    for _ in range(n_times):
        train_set, test_set = dataset.train_test_split(split_ratio)
        n_test_set_samples = len(test_set.attributes)
        model = DecisionTreeClassifier.train(train_set)
        evaluation = model.evaluate(test_set, positive_label, negative_label)
        evaluations.append(evaluation)

    accuracies = [evaluation.accuracy() for evaluation in evaluations]
    precisions = [evaluation.precision() for evaluation in evaluations]
    recalls = [evaluation.recall() for evaluation in evaluations]
    specificities = [evaluation.specificity() for evaluation in evaluations]

    tp = [evaluation.true_positives for evaluation in evaluations]
    tn = [evaluation.true_negatives for evaluation in evaluations]
    fp = [evaluation.false_positives for evaluation in evaluations]
    fn = [evaluation.false_negatives for evaluation in evaluations]

    print(f"Average values over {n_times} runs on {name} dataset")
    print(f"Number of samples in test set: {n_test_set_samples}")
    print(f"Accuracy:    {avg(accuracies) * 100:.2f}%")
    print(f"Precision:   {avg(precisions) * 100:.2f}%")
    print(f"Recall:      {avg(recalls) * 100:.2f}%")
    print(f"Specificity: {avg(specificities) * 100:.2f}%")
    print("")
    print(f"TP={avg(tp):<6.0f} FN={avg(fn):<6.0f}")
    print(f"FP={avg(fp):<6.0f} TN={avg(tn):<6.0f}")


def remove_missing_values(dataset: Dataset, empty_symbol: str) -> Dataset:
    new_dataset = Dataset()
    for idx in range(dataset.size()):
        attrs, label = dataset[idx]
        if empty_symbol not in attrs:
            new_dataset.add_row(attrs, label)

    return new_dataset
