from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.generalization import input_dimensions, model_data
from src.models import build_mic


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


def full_cohort_attribution(processed, model_state_path, params, config_module,
                            device, output_dir, n_steps=50):
    model = build_mic(input_dimensions(processed), params, config_module)
    model.load_state_dict(torch.load(
        model_state_path, map_location="cpu", weights_only=True
    ))
    model = model.to(device).eval()
    indices = np.arange(len(processed["participant_manifest"]))
    complete = model_data(processed, indices)
    data = {name:complete[name] for name in MODALITIES}
    baselines = {
        name:data[name].mean(dim=0,keepdim=True).numpy() for name in MODALITIES
    }
    with torch.no_grad():
        scores = MICClusterScore(model)(
            *(data[name].to(device) for name in MODALITIES)
        )
        targets = scores.argmax(dim=1).cpu().numpy()
    attributions, delta = calculate_ig(
        model,data,baselines,targets,device,config_module.BATCH_SIZE,n_steps
    )
    positive,absolute=modality_proportions(attributions)
    summary=processed["participant_manifest"][["row_index","id","benchmark_cluster"]].copy()
    summary["raw_predicted_cluster"]=targets; summary["convergence_delta"]=delta
    for index,name in enumerate(MODALITIES):
        summary[f"{name}_positive_proportion"]=positive[:,index]
        summary[f"{name}_absolute_proportion"]=absolute[:,index]
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output_dir/"participant_modality_summary.csv",index=False)
    for name in MODALITIES:
        pd.DataFrame(attributions[name],columns=processed[f"{name}_features"]).to_csv(
            output_dir/f"{name}_attributions.csv",index=False)
    return summary


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
