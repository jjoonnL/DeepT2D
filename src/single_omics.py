import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import apply_cluster_mapping, calculate_metrics, derive_cluster_mapping
from src.models import MICSingle
from src.training import set_deterministic_seed


MODALITIES = ("genotype", "proteome", "metabolite")
CLUSTER_LABELS = np.arange(4)


def _key(modality):
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    return f"input_{modality}"


def _data(processed, modality, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    return {
        "input": processed[_key(modality)].index_select(0, indices),
        "clinical": processed["output_clinical"].index_select(0, indices),
    }


def _loader(data, config_module, shuffle, seed, drop_last):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(data["input"], data["clinical"]),
        batch_size=config_module.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _optimizer(model, params, config_module):
    optimizer = AdamW(
        model.parameters(), lr=config_module.LEARNING_RATE,
        weight_decay=float(params["weight_decay"]),
    )
    scheduler = StepLR(
        optimizer, step_size=config_module.SCHEDULER_STEP_SIZE,
        gamma=config_module.SCHEDULER_GAMMA,
    )
    return optimizer, scheduler


def _validation_loss(model, data, config_module, device):
    loader = _loader(data, config_module, False, config_module.RANDOM_STATE, False)
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for inputs, clinical in loader:
            inputs, clinical = inputs.to(device), clinical.to(device)
            loss = F.mse_loss(model(inputs), clinical)
            total += float(loss.item()) * len(clinical)
            n += len(clinical)
    return total / n


def _train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for inputs, clinical in loader:
        inputs, clinical = inputs.to(device), clinical.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(inputs), clinical)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(clinical)
        n += len(clinical)
    return total / n


def _early_stop(input_dim, train, validation, params, config_module, device, seed):
    set_deterministic_seed(seed)
    model = MICSingle(input_dim, params).to(device)
    optimizer, scheduler = _optimizer(model, params, config_module)
    loader = _loader(train, config_module, True, seed, True)
    best_loss, best_state, best_epoch = np.inf, None, 0
    best_train_loss, waiting = np.nan, 0
    for epoch in range(1, config_module.MAX_EPOCHS + 1):
        train_loss = _train_epoch(model, loader, optimizer, device)
        validation_loss = _validation_loss(model, validation, config_module, device)
        if validation_loss < best_loss - config_module.EARLY_STOPPING_MIN_DELTA:
            best_loss, best_epoch, best_train_loss = validation_loss, epoch, train_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
        scheduler.step()
        if waiting >= config_module.EARLY_STOPPING_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch, "stopped_epoch": epoch,
        "best_train_loss": float(best_train_loss),
        "best_validation_loss": float(best_loss),
    }


def _fixed_epochs(input_dim, train, params, epochs, config_module, device, seed):
    set_deterministic_seed(seed)
    model = MICSingle(input_dim, params).to(device)
    optimizer, scheduler = _optimizer(model, params, config_module)
    loader = _loader(train, config_module, True, seed, True)
    loss = np.nan
    for _ in range(epochs):
        loss = _train_epoch(model, loader, optimizer, device)
        scheduler.step()
    return model, float(loss)


def _latent(model, data, config_module, device):
    loader = _loader(data, config_module, False, config_module.RANDOM_STATE, False)
    model.eval()
    values = []
    with torch.no_grad():
        for inputs, _ in loader:
            values.append(model.encode(inputs.to(device)).cpu().numpy())
    return np.concatenate(values)


def _fit_inner(processed, modality, train_idx, validation_idx, params,
               config_module, device, seed):
    train = _data(processed, modality, train_idx)
    validation = _data(processed, modality, validation_idx)
    model, info = _early_stop(
        train["input"].shape[1], train, validation, params,
        config_module, device, seed,
    )
    z_train = _latent(model, train, config_module, device)
    z_validation = _latent(model, validation, config_module, device)
    kmeans = KMeans(n_clusters=4, n_init=100,
                    random_state=config_module.RANDOM_STATE).fit(z_train)
    labels = processed["participant_manifest"]["benchmark_cluster"].to_numpy()
    mapping = derive_cluster_mapping(
        labels[train_idx], kmeans.predict(z_train), CLUSTER_LABELS
    )
    raw = kmeans.predict(z_validation)
    pred = apply_cluster_mapping(raw, mapping)
    metrics = {
        "val_accuracy": accuracy_score(labels[validation_idx], pred),
        "val_balanced_accuracy": balanced_accuracy_score(labels[validation_idx], pred),
        "val_ari": adjusted_rand_score(labels[validation_idx], raw),
    }
    return info, metrics


def run_inner_search(processed, modality, outer_fold, train_idx, config_module,
                     device, result_dir):
    labels = processed["participant_manifest"]["benchmark_cluster"].to_numpy()
    splitter = StratifiedKFold(
        n_splits=config_module.INNER_SPLITS, shuffle=True,
        random_state=config_module.INNER_RANDOM_STATE,
    )
    splits = list(splitter.split(np.zeros(len(train_idx)), labels[train_idx]))
    rows = []
    for params in ParameterGrid(config_module.PARAM_GRID):
        key = json.dumps(params, sort_keys=True)
        for inner_fold, (local_train, local_validation) in enumerate(splits, 1):
            info, metrics = _fit_inner(
                processed, modality, train_idx[local_train], train_idx[local_validation],
                params, config_module, device,
                config_module.RANDOM_STATE + outer_fold * 100 + inner_fold,
            )
            rows.append({"outer_fold":outer_fold,"inner_fold":inner_fold,
                         "parameter_key":key,**params,**metrics,**info})
    results = pd.DataFrame(rows)
    summary = results.groupby("parameter_key",as_index=False).agg(
        mean_val_accuracy=("val_accuracy","mean"),
        mean_val_balanced_accuracy=("val_balanced_accuracy","mean"),
        mean_val_ari=("val_ari","mean"),mean_best_epoch=("best_epoch","mean"),
        std_best_epoch=("best_epoch","std"),
        mean_stopped_epoch=("stopped_epoch","mean"),
    ).sort_values(["mean_val_accuracy","mean_val_ari"],ascending=False).reset_index(drop=True)
    summary["selected"] = False; summary.loc[0,"selected"] = True
    result_dir = Path(result_dir); result_dir.mkdir(parents=True,exist_ok=True)
    results.to_csv(result_dir/f"outer_{outer_fold:02d}_inner_search.csv",index=False)
    summary.to_csv(result_dir/f"outer_{outer_fold:02d}_inner_summary.csv",index=False)
    return json.loads(summary.loc[0,"parameter_key"]), max(1,int(np.rint(summary.loc[0,"mean_best_epoch"]))), summary


def run_outer_fold(processed, modality, outer_fold, config_module, device, output_root):
    result_dir = Path(output_root)/modality
    checkpoint_dir = result_dir/"checkpoints"; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    manifest = processed["participant_manifest"]
    labels = manifest["benchmark_cluster"].to_numpy(); folds=manifest["outer_fold"].to_numpy()
    train_idx=np.where(folds!=outer_fold)[0]; test_idx=np.where(folds==outer_fold)[0]
    params, epochs, summary = run_inner_search(
        processed,modality,outer_fold,train_idx,config_module,device,result_dir)
    train=_data(processed,modality,train_idx); test=_data(processed,modality,test_idx)
    model, final_loss = _fixed_epochs(train["input"].shape[1],train,params,epochs,
        config_module,device,config_module.RANDOM_STATE+outer_fold*1000+999)
    z_train=_latent(model,train,config_module,device); z_test=_latent(model,test,config_module,device)
    kmeans=KMeans(n_clusters=4,n_init=100,random_state=config_module.RANDOM_STATE).fit(z_train)
    raw_train=kmeans.predict(z_train); raw_test=kmeans.predict(z_test)
    mapping=derive_cluster_mapping(labels[train_idx],raw_train,CLUSTER_LABELS)
    pred=apply_cluster_mapping(raw_test,mapping)
    overall,class_rows=calculate_metrics(labels[test_idx],raw_test,pred,CLUSTER_LABELS,z_test)
    row={"modality":modality,"outer_fold":outer_fold,"train_n":len(train_idx),
         "test_n":len(test_idx),"best_params":json.dumps(params,sort_keys=True),
         "selected_epoch":epochs,"mean_inner_best_epoch":float(summary.loc[0,"mean_best_epoch"]),
         "final_train_loss":final_loss,**overall}
    pd.DataFrame([row]).to_csv(result_dir/f"outer_{outer_fold:02d}_metrics.csv",index=False)
    for item in class_rows: item.update({"modality":modality,"outer_fold":outer_fold})
    pd.DataFrame(class_rows).to_csv(result_dir/f"outer_{outer_fold:02d}_class_metrics.csv",index=False)
    predictions=pd.DataFrame({"row_index":test_idx,"id":manifest.iloc[test_idx]["id"].to_numpy(),
        "modality":modality,"outer_fold":outer_fold,"true_cluster":labels[test_idx],
        "raw_predicted_cluster":raw_test,"mapped_predicted_cluster":pred})
    predictions.to_csv(result_dir/f"outer_{outer_fold:02d}_predictions.csv",index=False)
    torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
        "best_params":params,"selected_epoch":epochs,"cluster_centers":kmeans.cluster_centers_,
        "cluster_mapping":mapping,"outer_train_idx":train_idx,"outer_test_idx":test_idx},
        checkpoint_dir/f"outer_{outer_fold:02d}.pt")
    return row
