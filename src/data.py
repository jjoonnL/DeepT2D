from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


def read_table(path):
    path = Path(path)
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=separator)


def load_input_data(genotype_path, proteome_path, metabolite_path, clinical_path):
    return {
        "genotype": read_table(genotype_path),
        "proteome": read_table(proteome_path),
        "metabolite": read_table(metabolite_path),
        "clinical": read_table(clinical_path),
    }


def _check_unique_ids(data, id_columns):
    for name, frame in data.items():
        id_column = id_columns[name]
        if id_column not in frame.columns:
            raise ValueError(f"Missing ID column '{id_column}' in {name}")
        if frame[id_column].duplicated().any():
            raise ValueError(f"Duplicate participant IDs in {name}")


def align_participants(data, id_columns):
    data = {name: frame.copy() for name, frame in data.items()}

    for name, frame in data.items():
        id_column = id_columns[name]
        frame[id_column] = frame[id_column].astype(str)

    _check_unique_ids(data, id_columns)

    common_ids = set.intersection(
        *[
            set(frame[id_columns[name]])
            for name, frame in data.items()
        ]
    )
    if not common_ids:
        raise ValueError("No participants are shared across all input files")

    id_order = sorted(common_ids)
    aligned = {}
    for name, frame in data.items():
        id_column = id_columns[name]
        aligned[name] = frame.set_index(id_column).loc[id_order].reset_index()

    for name, frame in aligned.items():
        if frame[id_columns[name]].tolist() != id_order:
            raise ValueError(f"Participant alignment failed for {name}")

    return aligned


def select_genotype_features(genotype, metadata_columns, clinical, sex_column):
    feature_columns = [
        column for column in genotype.columns
        if column not in set(metadata_columns)
    ]
    if not feature_columns:
        raise ValueError("No genotype features remain after removing metadata")

    sex = clinical[sex_column].to_numpy()
    usable = np.ones(len(feature_columns), dtype=bool)

    for sex_value in np.sort(pd.unique(sex)):
        matrix = genotype.loc[sex == sex_value, feature_columns]
        feature_std = matrix.std(axis=0, ddof=1).to_numpy()
        usable &= np.isfinite(feature_std) & (np.abs(feature_std) > 1e-12)

    selected = [
        feature for feature, keep in zip(feature_columns, usable)
        if keep
    ]
    if not selected:
        raise ValueError("No genotype features have variance in every sex group")

    return selected


def fit_sex_specific_scaler(matrix, sex):
    scaler = {}
    for sex_value in np.sort(pd.unique(sex)):
        mask = sex == sex_value
        sex_matrix = matrix[mask]
        if sex_matrix.shape[0] < 2:
            raise ValueError(f"Fewer than two participants for sex={sex_value}")

        mean = sex_matrix.mean(axis=0)
        std = sex_matrix.std(axis=0, ddof=1)
        zero_variance = (~np.isfinite(std)) | (np.abs(std) <= 1e-12)
        scale = std.copy()
        scale[zero_variance] = 1.0

        scaler[sex_value.item() if hasattr(sex_value, "item") else sex_value] = {
            "mean": mean,
            "scale": scale,
            "zero_variance": zero_variance,
            "n": int(mask.sum()),
        }

    return scaler


def apply_sex_specific_scaler(matrix, sex, scaler):
    transformed = np.empty_like(matrix, dtype=np.float64)

    for sex_value in np.sort(pd.unique(sex)):
        key = sex_value.item() if hasattr(sex_value, "item") else sex_value
        if key not in scaler:
            raise ValueError(f"Sex={sex_value} is absent from the scaler")
        mask = sex == sex_value
        transformed[mask] = (
            matrix[mask] - scaler[key]["mean"]
        ) / scaler[key]["scale"]

    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite values produced during scaling")

    return transformed.astype(np.float32)


def create_benchmark_labels(clinical_scaled, n_clusters, random_state):
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=100,
        random_state=random_state,
    )
    return model.fit_predict(clinical_scaled).astype(int)


def assign_outer_folds(labels, n_splits, random_state):
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    assignments = np.full(len(labels), -1, dtype=int)

    for fold, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        assignments[test_idx] = fold

    if np.any(assignments < 1):
        raise RuntimeError("Outer-fold assignment is incomplete")

    return assignments


def prepare_dataset(
    raw_data,
    clinical_targets,
    genotype_metadata_columns,
    id_columns,
    sex_column,
    n_clusters=4,
    random_state=214,
    outer_splits=10,
    outer_random_state=42,
):
    aligned = align_participants(raw_data, id_columns)
    clinical = aligned["clinical"]

    required_clinical = [sex_column, *clinical_targets]
    missing_clinical = [
        column for column in required_clinical
        if column not in clinical.columns
    ]
    if missing_clinical:
        raise ValueError(f"Missing clinical columns: {missing_clinical}")

    genotype_features = select_genotype_features(
        aligned["genotype"],
        genotype_metadata_columns,
        clinical,
        sex_column,
    )
    proteome_features = [
        column for column in aligned["proteome"].columns
        if column != id_columns["proteome"]
    ]
    metabolite_features = [
        column for column in aligned["metabolite"].columns
        if column != id_columns["metabolite"]
    ]

    feature_names = {
        "genotype": genotype_features,
        "proteome": proteome_features,
        "metabolite": metabolite_features,
        "clinical": list(clinical_targets),
    }
    raw_matrices = {
        "genotype": aligned["genotype"][genotype_features].to_numpy(dtype=np.float64),
        "proteome": aligned["proteome"][proteome_features].to_numpy(dtype=np.float64),
        "metabolite": aligned["metabolite"][metabolite_features].to_numpy(dtype=np.float64),
        "clinical": clinical[clinical_targets].to_numpy(dtype=np.float64),
    }

    for name, matrix in raw_matrices.items():
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite values in {name}")

    sex = clinical[sex_column].to_numpy()
    if pd.isna(sex).any():
        raise ValueError(f"Missing values in {sex_column}")
    if len(pd.unique(sex)) < 2:
        raise ValueError(f"Expected at least two groups in {sex_column}")

    scalers = {
        name: fit_sex_specific_scaler(matrix, sex)
        for name, matrix in raw_matrices.items()
    }
    scaled = {
        name: apply_sex_specific_scaler(matrix, sex, scalers[name])
        for name, matrix in raw_matrices.items()
    }

    benchmark_labels = create_benchmark_labels(
        scaled["clinical"], n_clusters, random_state
    )
    outer_fold = assign_outer_folds(
        benchmark_labels, outer_splits, outer_random_state
    )

    clinical_output = clinical.copy()
    if "kmeans_cluster" in clinical_output.columns:
        clinical_output["stored_file_cluster"] = clinical_output["kmeans_cluster"]
    clinical_output["benchmark_cluster"] = benchmark_labels
    clinical_output["kmeans_cluster"] = benchmark_labels

    id_order = clinical_output[id_columns["clinical"]].astype(str).tolist()
    participant_manifest = pd.DataFrame({
        "row_index": np.arange(len(clinical_output)),
        "id": id_order,
        "sex": sex,
        "benchmark_cluster": benchmark_labels,
        "outer_fold": outer_fold,
    })
    if "stored_file_cluster" in clinical_output.columns:
        participant_manifest["stored_file_cluster"] = clinical_output[
            "stored_file_cluster"
        ].to_numpy()

    return {
        "input_genotype": torch.from_numpy(scaled["genotype"]),
        "input_proteome": torch.from_numpy(scaled["proteome"]),
        "input_metabolite": torch.from_numpy(scaled["metabolite"]),
        "output_clinical": torch.from_numpy(scaled["clinical"]),
        "clinical_df": clinical_output,
        "participant_manifest": participant_manifest,
        "genotype_features": genotype_features,
        "proteome_features": proteome_features,
        "metabolite_features": metabolite_features,
        "clinical_targets": list(clinical_targets),
        "preprocessing": {
            "method": "sex-specific cohort-level scaling",
            "scalers": scalers,
        },
    }


def save_preprocessed_data(processed_data, output_dir, dataset_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(processed_data, dataset_path)
    processed_data["participant_manifest"].to_csv(
        output_dir / "participant_manifest.csv", index=False
    )

    feature_manifests = {
        "genotype_features.csv": processed_data["genotype_features"],
        "proteome_features.csv": processed_data["proteome_features"],
        "metabolite_features.csv": processed_data["metabolite_features"],
        "clinical_targets.csv": processed_data["clinical_targets"],
    }
    for filename, features in feature_manifests.items():
        pd.DataFrame({"feature": features}).to_csv(
            output_dir / filename, index=False
        )


def summarize_processed_data(processed_data):
    manifest = processed_data["participant_manifest"]
    return {
        "participants": len(manifest),
        "genotype_features": len(processed_data["genotype_features"]),
        "proteome_features": len(processed_data["proteome_features"]),
        "metabolite_features": len(processed_data["metabolite_features"]),
        "clinical_targets": len(processed_data["clinical_targets"]),
        "benchmark_counts": manifest["benchmark_cluster"].value_counts().sort_index().to_dict(),
        "outer_fold_counts": manifest["outer_fold"].value_counts().sort_index().to_dict(),
    }
