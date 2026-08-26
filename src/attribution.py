from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from src.full_cohort import clinical_subtype_mapping
from src.generalization import input_dimensions, model_data
from src.models import build_mic
from src.training import extract_latent


MODALITIES = ("genotype", "proteome", "metabolite")


class MICClusterScore(nn.Module):
    def __init__(self, model, alpha=1.0):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, genotype, proteome, metabolite):
        latent = self.model.encode(genotype, proteome, metabolite)
        return self.model.soft_cluster_assignment(latent, alpha=self.alpha)


def calculate_ig(model, data, baselines, targets, device, batch_size=64,
                 n_steps=50):
    from captum.attr import IntegratedGradients

    wrapper = MICClusterScore(model).to(device).eval()
    integrated_gradients = IntegratedGradients(wrapper)
    attribution_blocks = {name: [] for name in MODALITIES}
    deltas = []
    baseline_tensors = tuple(
        torch.as_tensor(baselines[name], dtype=torch.float32, device=device)
        for name in MODALITIES
    )

    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        inputs = tuple(data[name][start:stop].to(device) for name in MODALITIES)
        batch_baselines = tuple(
            baseline.expand(stop-start, -1) for baseline in baseline_tensors
        )
        target = torch.as_tensor(targets[start:stop], dtype=torch.long, device=device)
        attributions, delta = integrated_gradients.attribute(
            inputs, baselines=batch_baselines, target=target,
            n_steps=n_steps, return_convergence_delta=True,
        )
        for name, values in zip(MODALITIES, attributions):
            attribution_blocks[name].append(values.detach().cpu().numpy())
        deltas.append(delta.detach().cpu().numpy())
    return {
        name: np.concatenate(blocks) for name, blocks in attribution_blocks.items()
    }, np.concatenate(deltas)


def modality_proportions(attributions):
    positive = np.column_stack([
        np.maximum(attributions[name], 0).sum(axis=1) for name in MODALITIES
    ])
    absolute = np.column_stack([
        np.abs(attributions[name]).sum(axis=1) for name in MODALITIES
    ])
    positive_denominator = np.where(positive.sum(axis=1)==0, 1, positive.sum(axis=1))
    absolute_denominator = np.where(absolute.sum(axis=1)==0, 1, absolute.sum(axis=1))
    return positive/positive_denominator[:,None], absolute/absolute_denominator[:,None]


def full_cohort_attribution(processed, params, config_module, device,
                            output_dir, n_steps=50, top_k=5):
    full_cohort_dir = Path(config_module.OUTPUT_DIR) / "full_cohort"
    manifest = pd.read_csv(full_cohort_dir / "model_manifest.csv")
    run_metrics = pd.read_csv(full_cohort_dir / "full_cohort_run_metrics.csv")
    valid = run_metrics.loc[run_metrics["included_in_vote"], ["run", "accuracy", "ari"]]
    valid = valid.merge(manifest[["run", "seed", "model_path"]], on="run")
    if valid.empty:
        raise RuntimeError("No full-cohort models met the attribution criteria")

    indices = np.arange(len(processed["participant_manifest"]))
    complete = model_data(processed, indices)
    data = {name: complete[name] for name in MODALITIES}
    baselines = {
        name: data[name].mean(dim=0, keepdim=True).numpy()
        for name in MODALITIES
    }
    clinical = processed["clinical_df"].copy()
    subtype_vectors = {
        subtype: {name: [] for name in MODALITIES}
        for subtype in ("SIRD", "SIDD", "MOD", "MARD")
    }
    convergence_rows = []
    allow_mapping_fallback = "sample_data" in Path(
        config_module.CLINICAL_PATH
    ).parts

    for row in tqdm(
        valid.itertuples(index=False),
        total=len(valid),
        desc="Full-cohort IG models",
    ):
        model = build_mic(input_dimensions(processed), params, config_module)
        state = torch.load(
            full_cohort_dir / row.model_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model = model.to(device).eval()

        latent = extract_latent(model, complete, config_module, device)
        kmeans = KMeans(
            n_clusters=config_module.NUM_CLUSTERS,
            init="k-means++",
            n_init=100,
            random_state=config_module.RANDOM_STATE,
        ).fit(latent)
        clusters = kmeans.labels_
        mapping = clinical_subtype_mapping(
            clinical, clusters, allow_fallback=allow_mapping_fallback
        )
        if mapping is None:
            raise RuntimeError(f"Subtype mapping failed for included run {row.run}")
        model.centroid.data.copy_(torch.as_tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        ))

        for cluster, subtype in mapping.items():
            subtype_indices = np.flatnonzero(clusters == cluster)
            subtype_data = {
                name: data[name][subtype_indices] for name in MODALITIES
            }
            targets = np.full(len(subtype_indices), cluster, dtype=int)
            attributions, delta = calculate_ig(
                model, subtype_data, baselines, targets, device,
                config_module.BATCH_SIZE, n_steps,
            )
            for name in MODALITIES:
                subtype_vectors[subtype][name].append(
                    attributions[name].mean(axis=0)
                )
            convergence_rows.append({
                "run": int(row.run),
                "subtype": subtype,
                "assigned_n": int(len(subtype_indices)),
                "mean_convergence_delta": float(np.mean(delta)),
                "max_absolute_convergence_delta": float(np.max(np.abs(delta))),
            })

    mean_vectors = {}
    modality_rows = []
    feature_rows = []
    top_rows = []
    for subtype, modality_vectors in subtype_vectors.items():
        mean_vectors[subtype] = {
            name: np.mean(modality_vectors[name], axis=0)
            for name in MODALITIES
        }
        positive_sums = {
            name: float(np.maximum(mean_vectors[subtype][name], 0).sum())
            for name in MODALITIES
        }
        denominator = sum(positive_sums.values()) or 1.0
        for name in MODALITIES:
            values = mean_vectors[subtype][name]
            features = processed[f"{name}_features"]
            modality_rows.append({
                "subtype": subtype,
                "modality": name,
                "contribution_percent": 100 * positive_sums[name] / denominator,
                "model_n": len(modality_vectors[name]),
            })
            order = np.argsort(values)[::-1]
            for rank, feature_index in enumerate(order, start=1):
                feature_rows.append({
                    "subtype": subtype,
                    "modality": name,
                    "feature": features[feature_index],
                    "mean_attribution": float(values[feature_index]),
                    "rank": rank,
                })
                if rank <= top_k:
                    top_rows.append(feature_rows[-1].copy())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modality_summary = pd.DataFrame(modality_rows)
    feature_summary = pd.DataFrame(feature_rows)
    top_features = pd.DataFrame(top_rows)
    modality_summary.to_csv(output_dir / "modality_summary.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_summary.csv", index=False)
    top_features.to_csv(output_dir / "top_features.csv", index=False)
    valid.to_csv(output_dir / "included_models.csv", index=False)
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "convergence_summary.csv", index=False
    )
    return modality_summary, top_features


def heldout_fold(processed, checkpoint_path, outer_fold, config_module, device,
                 output_dir, n_steps=50):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    test_idx = np.asarray(checkpoint["outer_test_idx"], dtype=int)
    train_idx = np.asarray(checkpoint["outer_train_idx"], dtype=int)
    model = build_mic(input_dimensions(processed), checkpoint["best_params"], config_module)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.centroid.data.copy_(torch.as_tensor(checkpoint["cluster_centers"],dtype=torch.float32))
    model = model.to(device)
    test_data = model_data(processed, test_idx)
    train_data = model_data(processed, train_idx)
    data = {name:test_data[name] for name in MODALITIES}
    baselines = {
        name:train_data[name].mean(dim=0,keepdim=True).numpy() for name in MODALITIES
    }
    output_dir = Path(output_dir)
    prediction_path = output_dir.parent/"nested_cv"/f"outer_{outer_fold:02d}_predictions.csv"
    if prediction_path.exists():
        prediction = pd.read_csv(prediction_path)
        targets = prediction["raw_predicted_cluster"].to_numpy(dtype=int)
        mapped = prediction["mapped_predicted_cluster"].to_numpy(dtype=int)
    else:
        centers = np.asarray(checkpoint["cluster_centers"])
        with torch.no_grad():
            latent = model.encode(*(data[name].to(device) for name in MODALITIES)).cpu().numpy()
        targets = np.argmin(((latent[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
        mapped = np.asarray([checkpoint["cluster_mapping"][int(x)] for x in targets])
    attributions, delta = calculate_ig(
        model,data,baselines,targets,device,config_module.BATCH_SIZE,n_steps
    )
    positive, absolute = modality_proportions(attributions)
    manifest = processed["participant_manifest"].iloc[test_idx]
    summary = pd.DataFrame({
        "row_index":test_idx,"id":manifest["id"].to_numpy(),"outer_fold":outer_fold,
        "true_cluster":manifest["benchmark_cluster"].to_numpy(),
        "raw_predicted_cluster":targets,"mapped_predicted_cluster":mapped,
        "convergence_delta":delta,
    })
    for index,name in enumerate(MODALITIES):
        summary[f"{name}_positive_proportion"] = positive[:,index]
        summary[f"{name}_absolute_proportion"] = absolute[:,index]
        frame=pd.DataFrame(attributions[name],columns=processed[f"{name}_features"])
        frame.insert(0,"row_index",test_idx); frame.insert(1,"outer_fold",outer_fold)
        output_dir.mkdir(parents=True,exist_ok=True)
        frame.to_csv(output_dir/f"outer_{outer_fold:02d}_{name}_attributions.csv",index=False)
    summary.to_csv(output_dir/f"outer_{outer_fold:02d}_participant_summary.csv",index=False)
    return summary


def aggregate_heldout(processed, output_dir, top_k=5, recurrence_min=5):
    output_dir = Path(output_dir)
    participant = pd.concat([
        pd.read_csv(path) for path in sorted(output_dir.glob("outer_*_participant_summary.csv"))
    ],ignore_index=True)
    modality_rows=[]; feature_rows=[]
    for subtype,group in participant.groupby("mapped_predicted_cluster"):
        for name in MODALITIES:
            modality_rows.append({"predicted_subtype":int(subtype),"modality":name,
                "assigned_n":len(group),
                "mean_positive_proportion":group[f"{name}_positive_proportion"].mean(),
                "sd_positive_proportion":group[f"{name}_positive_proportion"].std()})
    for name in MODALITIES:
        features=processed[f"{name}_features"]
        frames=[]
        for path in sorted(output_dir.glob(f"outer_*_{name}_attributions.csv")):
            frame=pd.read_csv(path); frames.append(frame)
        values=pd.concat(frames,ignore_index=True).merge(
            participant[["row_index","mapped_predicted_cluster"]],on="row_index"
        )
        for subtype,group in values.groupby("mapped_predicted_cluster"):
            positive=np.maximum(group[features].to_numpy(),0).mean(axis=0)
            absolute=np.abs(group[features].to_numpy()).mean(axis=0)
            correlation=spearmanr(positive,absolute).statistic
            for index,feature in enumerate(features):
                feature_rows.append({"predicted_subtype":int(subtype),"modality":name,
                    "feature":feature,"mean_positive":positive[index],
                    "mean_absolute":absolute[index],
                    "positive_absolute_spearman":correlation})
    modality_summary=pd.DataFrame(modality_rows)
    feature_summary=pd.DataFrame(feature_rows)
    recurrence=[]
    for name in MODALITIES:
        features=processed[f"{name}_features"]
        frames=[]
        for path in sorted(output_dir.glob(f"outer_*_{name}_attributions.csv")):
            fold=int(path.name.split("_")[1]); frame=pd.read_csv(path).merge(
                participant[["row_index","mapped_predicted_cluster"]],on="row_index")
            for subtype,group in frame.groupby("mapped_predicted_cluster"):
                scores=np.maximum(group[features].to_numpy(),0).mean(axis=0)
                for index in np.argsort(scores)[-top_k:]:
                    frames.append((int(subtype),features[index],fold))
        recurrence_frame=pd.DataFrame(frames,columns=["predicted_subtype","feature","outer_fold"])
        if not recurrence_frame.empty:
            counts=recurrence_frame.groupby(["predicted_subtype","feature"]).size().reset_index(name="folds")
            counts["modality"]=name
            recurrence.append(counts[counts["folds"]>=recurrence_min])
    recurrent=pd.concat(recurrence,ignore_index=True) if recurrence else pd.DataFrame()
    modality_summary.to_csv(output_dir/"modality_summary.csv",index=False)
    feature_summary.to_csv(output_dir/"feature_summary.csv",index=False)
    recurrent.to_csv(output_dir/"recurrent_top_features.csv",index=False)
    return modality_summary,feature_summary,recurrent
