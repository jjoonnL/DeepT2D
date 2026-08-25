import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.evaluation import (
    apply_cluster_mapping,
    calculate_metrics,
    derive_cluster_mapping,
)
from src.training import extract_latent, train_fixed_epochs, train_with_early_stopping


CLUSTER_LABELS = np.arange(4)


def parameter_key(params):
    return json.dumps(params, sort_keys=True)


def model_data(processed_data, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        "genotype": processed_data["input_genotype"].index_select(0, indices),
        "proteome": processed_data["input_proteome"].index_select(0, indices),
        "metabolite": processed_data["input_metabolite"].index_select(0, indices),
        "clinical": processed_data["output_clinical"].index_select(0, indices),
    }


def input_dimensions(processed_data):
    return {
        "genotype": processed_data["input_genotype"].shape[1],
        "proteome": processed_data["input_proteome"].shape[1],
        "metabolite": processed_data["input_metabolite"].shape[1],
    }


def fit_and_evaluate_inner(processed_data, train_idx, validation_idx, params,
                           config_module, device, seed):
    train_data = model_data(processed_data, train_idx)
    validation_data = model_data(processed_data, validation_idx)
    dimensions = input_dimensions(processed_data)
    labels = processed_data["participant_manifest"]["benchmark_cluster"].to_numpy()

    model, training_info = train_with_early_stopping(
        train_data,
        validation_data,
        dimensions,
        params,
        config_module,
        device,
        seed,
    )
    train_latent = extract_latent(model, train_data, config_module, device)
    validation_latent = extract_latent(
        model, validation_data, config_module, device
    )
    kmeans = KMeans(
        n_clusters=config_module.NUM_CLUSTERS,
        n_init=100,
        random_state=config_module.RANDOM_STATE,
    ).fit(train_latent)
    train_clusters = kmeans.predict(train_latent)
    validation_clusters = kmeans.predict(validation_latent)
    mapping = derive_cluster_mapping(
        labels[train_idx], train_clusters, CLUSTER_LABELS
    )
    validation_predictions = apply_cluster_mapping(
        validation_clusters, mapping
    )
    metrics = {
        "val_accuracy": accuracy_score(
            labels[validation_idx], validation_predictions
        ),
        "val_balanced_accuracy": balanced_accuracy_score(
            labels[validation_idx], validation_predictions
        ),
        "val_ari": adjusted_rand_score(
            labels[validation_idx], validation_clusters
        ),
    }
    return model, training_info, metrics


def run_inner_search(processed_data, outer_fold, outer_train_idx, config_module,
                     device, output_dir=None, parameter_grid=None):
    labels = processed_data["participant_manifest"]["benchmark_cluster"].to_numpy()
    inner_cv = StratifiedKFold(
        n_splits=config_module.INNER_SPLITS,
        shuffle=True,
        random_state=config_module.INNER_RANDOM_STATE,
    )
    splits = list(inner_cv.split(
        np.zeros(len(outer_train_idx)), labels[outer_train_idx]
    ))
    grid = config_module.PARAM_GRID if parameter_grid is None else parameter_grid
    rows = []
    params_by_key = {}

    for params in ParameterGrid(grid):
        key = parameter_key(params)
        params_by_key[key] = dict(params)
        for inner_fold, (train_local, validation_local) in enumerate(splits, 1):
            train_idx = outer_train_idx[train_local]
            validation_idx = outer_train_idx[validation_local]
            seed = config_module.RANDOM_STATE + outer_fold * 100 + inner_fold
            model, training_info, metrics = fit_and_evaluate_inner(
                processed_data,
                train_idx,
                validation_idx,
                params,
                config_module,
                device,
                seed,
            )
            rows.append({
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "parameter_key": key,
                **dict(params),
                **metrics,
                "best_epoch": training_info["best_epoch"],
                "stopped_epoch": training_info["stopped_epoch"],
                "best_train_loss": training_info["best_train_loss"],
                "best_validation_loss": training_info["best_validation_loss"],
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    summary = (
        results.groupby("parameter_key", as_index=False)
        .agg(
            mean_val_accuracy=("val_accuracy", "mean"),
            mean_val_balanced_accuracy=("val_balanced_accuracy", "mean"),
            mean_val_ari=("val_ari", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            std_best_epoch=("best_epoch", "std"),
            mean_stopped_epoch=("stopped_epoch", "mean"),
        )
        .sort_values(
            ["mean_val_accuracy", "mean_val_ari"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    best_key = summary.loc[0, "parameter_key"]
    selected_epoch = max(1, int(np.rint(summary.loc[0, "mean_best_epoch"])))
    summary["selected"] = False
    summary.loc[0, "selected"] = True

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            output_dir / f"outer_{outer_fold:02d}_inner_search.csv", index=False
        )
        summary.to_csv(
            output_dir / f"outer_{outer_fold:02d}_inner_summary.csv", index=False
        )
    return params_by_key[best_key], selected_epoch, results, summary


def run_outer_fold(processed_data, outer_fold, config_module, device,
                   output_dir):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manifest = processed_data["participant_manifest"]
    labels = manifest["benchmark_cluster"].to_numpy()
    fold_assignment = manifest["outer_fold"].to_numpy()
    test_idx = np.where(fold_assignment == outer_fold)[0]
    train_idx = np.where(fold_assignment != outer_fold)[0]

    best_params, selected_epoch, _, inner_summary = run_inner_search(
        processed_data,
        outer_fold,
        train_idx,
        config_module,
        device,
        output_dir,
    )
    train_data = model_data(processed_data, train_idx)
    test_data = model_data(processed_data, test_idx)
    model, training_info = train_fixed_epochs(
        train_data,
        input_dimensions(processed_data),
        best_params,
        selected_epoch,
        config_module,
        device,
        config_module.RANDOM_STATE + outer_fold * 1000 + 999,
    )
    train_latent = extract_latent(model, train_data, config_module, device)
    test_latent = extract_latent(model, test_data, config_module, device)
    kmeans = KMeans(
        n_clusters=config_module.NUM_CLUSTERS,
        n_init=100,
        random_state=config_module.RANDOM_STATE,
    ).fit(train_latent)
    train_clusters = kmeans.predict(train_latent)
    test_clusters = kmeans.predict(test_latent)
    mapping = derive_cluster_mapping(
        labels[train_idx], train_clusters, CLUSTER_LABELS
    )
    test_predictions = apply_cluster_mapping(test_clusters, mapping)
    overall, class_rows = calculate_metrics(
        labels[test_idx], test_clusters, test_predictions, CLUSTER_LABELS,
        latent=test_latent,
    )

    predictions = pd.DataFrame({
        "row_index": test_idx,
        "id": manifest.iloc[test_idx]["id"].to_numpy(),
        "outer_fold": outer_fold,
        "true_cluster": labels[test_idx],
        "raw_predicted_cluster": test_clusters,
        "mapped_predicted_cluster": test_predictions,
    })
    predictions.to_csv(
        output_dir / f"outer_{outer_fold:02d}_predictions.csv", index=False
    )
    for row in class_rows:
        row["outer_fold"] = outer_fold
    pd.DataFrame(class_rows).to_csv(
        output_dir / f"outer_{outer_fold:02d}_class_metrics.csv", index=False
    )

    overall_row = {
        "outer_fold": outer_fold,
        "train_n": len(train_idx),
        "fit_n": len(train_idx),
        "test_n": len(test_idx),
        "best_params": parameter_key(best_params),
        "epochs": training_info["epochs"],
        "selected_epoch": selected_epoch,
        "mean_inner_best_epoch": float(inner_summary.loc[0, "mean_best_epoch"]),
        **overall,
    }
    pd.DataFrame([overall_row]).to_csv(
        output_dir / f"outer_{outer_fold:02d}_metrics.csv", index=False
    )
    torch.save({
        "outer_fold": outer_fold,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "best_params": best_params,
        "training_info": {
            key: value for key, value in training_info.items() if key != "history"
        },
        "selected_epoch": selected_epoch,
        "mean_inner_best_epoch": float(inner_summary.loc[0, "mean_best_epoch"]),
        "cluster_centers": kmeans.cluster_centers_,
        "cluster_mapping": mapping,
        "preprocessing": processed_data.get("preprocessing"),
        "outer_train_idx": train_idx,
        "outer_test_idx": test_idx,
    }, checkpoint_dir / f"outer_{outer_fold:02d}.pt")
    return overall_row, class_rows, predictions


def aggregate_outer_folds(output_dir):
    output_dir = Path(output_dir)
    metric_files = sorted(output_dir.glob("outer_[0-9][0-9]_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError("No completed outer-fold metrics were found")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    classes = pd.concat([
        pd.read_csv(output_dir / path.name.replace("_metrics", "_class_metrics"))
        for path in metric_files
    ], ignore_index=True)
    predictions = pd.concat([
        pd.read_csv(output_dir / path.name.replace("_metrics", "_predictions"))
        for path in metric_files
    ], ignore_index=True).sort_values("row_index")
    metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    classes.to_csv(output_dir / "class_metrics.csv", index=False)
    predictions.to_csv(output_dir / "participant_predictions.csv", index=False)
    return metrics, classes, predictions


def analysis_configuration(config_module):
    return {
        "base_seed": config_module.RANDOM_STATE,
        "outer_splits": config_module.OUTER_SPLITS,
        "inner_splits": config_module.INNER_SPLITS,
        "outer_random_state": config_module.OUTER_RANDOM_STATE,
        "inner_random_state": config_module.INNER_RANDOM_STATE,
        "encoder_dim": config_module.ENCODER_DIMS,
        "hidden_dim": config_module.INTEGRATION_DIMS,
        "latent_dim": config_module.LATENT_DIM,
        "learning_rate": config_module.LEARNING_RATE,
        "batch_size": config_module.BATCH_SIZE,
        "epoch_selection": "mean inner-fold best epoch",
        "max_epochs": config_module.MAX_EPOCHS,
        "patience": config_module.EARLY_STOPPING_PATIENCE,
        "min_delta": config_module.EARLY_STOPPING_MIN_DELTA,
        "preprocessing": "sex-specific cohort-level global scaling",
        "scheduler": {
            "name": "StepLR",
            "step_size": config_module.SCHEDULER_STEP_SIZE,
            "gamma": config_module.SCHEDULER_GAMMA,
        },
        "parameter_grid": config_module.PARAM_GRID,
    }
