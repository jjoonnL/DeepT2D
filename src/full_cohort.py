from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

from src.evaluation import derive_cluster_mapping
from src.generalization import input_dimensions, model_data
from src.models import build_mic
from src.training import extract_latent, train_fixed_epochs


SUBTYPES = ("SIRD", "SIDD", "MOD", "MARD")
CLUSTER_LABELS = np.arange(4)


def final_parameters(config_module):
    return {
        "decoder_dim": list(config_module.DECODER_DIMS),
        "dropout": config_module.DROPOUT,
        "weight_decay": config_module.WEIGHT_DECAY,
    }


def train_runs(processed, config_module, device, output_dir, runs=None):
    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    runs = config_module.NUM_RUNS if runs is None else runs
    indices = np.arange(len(processed["participant_manifest"]))
    data = model_data(processed, indices)
    dimensions = input_dimensions(processed)
    rows = []

    for run in tqdm(range(runs), desc="Training full-cohort models"):
        seed = 100 + run
        model, info = train_fixed_epochs(
            data, dimensions, final_parameters(config_module),
            config_module.EPOCHS, config_module, device, seed,
        )
        path = model_dir / f"mic_run_{run:03d}.pth"
        torch.save(model.state_dict(), path)
        rows.append({
            "run": run, "seed": seed, "model_path": str(path.relative_to(output_dir)),
            "epochs": config_module.EPOCHS,
            "final_train_loss": info["final_train_loss"],
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "model_manifest.csv", index=False)
    return manifest


def clinical_subtype_mapping(clinical, clusters):
    values = clinical.copy()
    values["mic_cluster"] = clusters
    features = ("HOMA_IR", "hba1c", "bmi", "age_at_diagnosis")
    means = values.groupby("mic_cluster")[list(features)].mean()
    mapping = {}
    used = set()
    for feature, subtype in zip(features, SUBTYPES):
        cluster = int(means[feature].idxmax())
        if cluster in used:
            return None
        mapping[cluster] = subtype
        used.add(cluster)
    return mapping


def evaluate_runs(processed, config_module, device, output_dir,
                  accuracy_threshold=0.75):
    output_dir = Path(output_dir)
    manifest = pd.read_csv(output_dir / "model_manifest.csv")
    expected_runs = list(range(config_module.NUM_RUNS))
    if manifest["run"].tolist() != expected_runs:
        raise ValueError("Model manifest is incomplete or out of order")

    participant_manifest = processed["participant_manifest"]
    true_labels = participant_manifest["benchmark_cluster"].to_numpy()
    clinical = processed["clinical_df"].copy()
    indices = np.arange(len(participant_manifest))
    data = model_data(processed, indices)
    dimensions = input_dimensions(processed)
    prediction_columns = {}
    run_rows = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Evaluating full-cohort models",
    ):
        model = build_mic(dimensions, final_parameters(config_module), config_module).to(device)
        state = torch.load(output_dir / row.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        latent = extract_latent(model, data, config_module, device)
        clusters = KMeans(
            n_clusters=config_module.NUM_CLUSTERS, n_init=100,
            random_state=config_module.RANDOM_STATE,
        ).fit_predict(latent)
        numeric_mapping = derive_cluster_mapping(true_labels, clusters, CLUSTER_LABELS)
        aligned = np.asarray([numeric_mapping[int(x)] for x in clusters])
        accuracy = float(np.mean(aligned == true_labels))
        ari = adjusted_rand_score(true_labels, clusters)
        subtype_mapping = clinical_subtype_mapping(clinical, clusters)
        valid = subtype_mapping is not None and accuracy >= accuracy_threshold
        if subtype_mapping is not None:
            prediction_columns[f"run_{row.run:03d}"] = pd.Series(clusters).map(subtype_mapping)
        run_rows.append({
            "run": row.run, "seed": row.seed, "accuracy": accuracy,
            "ari": ari, "mapping_succeeded": subtype_mapping is not None,
            "included_in_vote": valid,
        })

    run_metrics = pd.DataFrame(run_rows)
    run_metrics.to_csv(output_dir / "full_cohort_run_metrics.csv", index=False)
    valid_columns = [
        f"run_{run:03d}" for run in run_metrics.loc[
            run_metrics["included_in_vote"], "run"
        ]
    ]
    if not valid_columns:
        raise RuntimeError("No run met the majority-vote inclusion criteria")
    predictions = pd.DataFrame(prediction_columns)
    final_labels = predictions[valid_columns].mode(axis=1)[0]
    output = clinical.copy()
    output["mic_cluster"] = final_labels
    output.to_csv(output_dir / "final_clinical_data_with_labels.csv", index=False)
    return run_metrics, output
