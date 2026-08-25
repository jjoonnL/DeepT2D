import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


def derive_cluster_mapping(training_labels, training_clusters, cluster_labels):
    contingency = confusion_matrix(
        training_labels, training_clusters, labels=cluster_labels
    )
    true_rows, cluster_columns = linear_sum_assignment(
        contingency, maximize=True
    )
    mapping = {
        int(cluster_labels[cluster_column]): int(cluster_labels[true_row])
        for true_row, cluster_column in zip(true_rows, cluster_columns)
    }
    if set(mapping) != set(cluster_labels):
        raise ValueError("Incomplete cluster-to-subtype mapping")
    return mapping


def apply_cluster_mapping(clusters, mapping):
    return np.asarray([mapping[int(cluster)] for cluster in clusters], dtype=int)


def calculate_metrics(true_labels, raw_clusters, mapped_predictions,
                      cluster_labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        mapped_predictions,
        labels=cluster_labels,
        zero_division=0,
    )
    overall = {
        "accuracy": accuracy_score(true_labels, mapped_predictions),
        "balanced_accuracy": balanced_accuracy_score(
            true_labels, mapped_predictions
        ),
        "ari": adjusted_rand_score(true_labels, raw_clusters),
        "nmi": normalized_mutual_info_score(true_labels, raw_clusters),
        "ami": adjusted_mutual_info_score(true_labels, raw_clusters),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    class_rows = [
        {
            "class_label": int(label),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(cluster_labels)
    ]
    return overall, class_rows
