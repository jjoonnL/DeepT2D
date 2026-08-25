import json
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from src.evaluation import apply_cluster_mapping, calculate_metrics, derive_cluster_mapping


FINAL_MODELS = ("pca_kmeans", "mofa_kmeans", "elastic_net")
CLUSTER_LABELS = np.arange(4)
ELASTIC_NET_GRID = list(ParameterGrid({
    "C": [0.01, 0.1, 1.0],
    "l1_ratio": [0.1, 0.9],
    "class_weight": [None, "balanced"],
}))
MOFA_VIEWS = ("genotype", "proteome", "metabolite")
MOFA_FACTORS = 16


def multi_omics_matrix(processed):
    return np.concatenate([
        processed[f"input_{view}"].numpy() for view in MOFA_VIEWS
    ], axis=1)


def _classifier(params, seed):
    return LogisticRegression(
        penalty="elasticnet", solver="saga", C=float(params["C"]),
        l1_ratio=float(params["l1_ratio"]), class_weight=params["class_weight"],
        max_iter=5000, tol=1e-4, random_state=seed,
    )


def _evaluate_elastic(params, matrix, labels, train_idx, outer_fold,
                      inner_splits, base_seed):
    metrics = []
    for inner_fold, (local_train, local_validation) in enumerate(inner_splits, 1):
        fit_idx, validation_idx = train_idx[local_train], train_idx[local_validation]
        model = _classifier(params, base_seed + 1000 * outer_fold + inner_fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(matrix[fit_idx], labels[fit_idx])
        prediction = model.predict(matrix[validation_idx])
        metrics.append({
            "accuracy": accuracy_score(labels[validation_idx], prediction),
            "balanced_accuracy": balanced_accuracy_score(labels[validation_idx], prediction),
            "ari": adjusted_rand_score(labels[validation_idx], prediction),
        })
    return {
        "model":"elastic_net","outer_fold":outer_fold,
        "parameter_key":json.dumps(params,sort_keys=True),
        "mean_val_accuracy":np.mean([x["accuracy"] for x in metrics]),
        "mean_val_balanced_accuracy":np.mean([x["balanced_accuracy"] for x in metrics]),
        "mean_val_ari":np.mean([x["ari"] for x in metrics]),
    }


def elastic_inner_search(matrix, labels, train_idx, outer_fold, config_module,
                         n_jobs=8):
    splitter = StratifiedKFold(
        n_splits=config_module.INNER_SPLITS, shuffle=True,
        random_state=config_module.INNER_RANDOM_STATE,
    )
    splits = list(splitter.split(np.zeros(len(train_idx)), labels[train_idx]))
    with parallel_backend("loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs)(delayed(_evaluate_elastic)(
            params,matrix,labels,train_idx,outer_fold,splits,config_module.RANDOM_STATE
        ) for params in ELASTIC_NET_GRID)
    search = pd.DataFrame(rows).sort_values(
        ["mean_val_accuracy","mean_val_balanced_accuracy","mean_val_ari"],
        ascending=False,
    ).reset_index(drop=True)
    return json.loads(search.loc[0,"parameter_key"]), search


def collapse_tau_expectation(expectation, n_features):
    tau = np.asarray(expectation, dtype=np.float64)
    vector = tau if tau.ndim == 1 else np.nanmedian(tau, axis=0)
    if vector.shape != (n_features,) or not np.isfinite(vector).all() or np.any(vector <= 0):
        raise ValueError("Invalid MOFA feature precision")
    return vector


def project_gaussian_mofa(views, weights, second_moments, precisions):
    n_samples, n_factors = views[0].shape[0], weights[0].shape[1]
    posterior_precision = np.eye(n_factors)
    linear_term = np.zeros((n_samples, n_factors))
    for matrix, weight, second_moment, tau in zip(
        views, weights, second_moments, precisions
    ):
        weighted_loading = tau[:, None] * weight
        posterior_precision += weight.T @ weighted_loading
        loading_variance = np.maximum(second_moment - weight ** 2, 0.0)
        posterior_precision += np.diag(np.sum(tau[:, None] * loading_variance, axis=0))
        linear_term += (matrix * tau[None, :]) @ weight
    scores = np.linalg.solve(posterior_precision, linear_term.T).T
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite held-out MOFA factor score")
    return scores


def factor_projection_qc(reference, projected):
    correlations = [
        np.corrcoef(reference[:, index], projected[:, index])[0, 1]
        for index in range(reference.shape[1])
        if np.std(reference[:, index]) > 1e-10 and np.std(projected[:, index]) > 1e-10
    ]
    normalized_rmse = np.linalg.norm(reference-projected)/max(np.linalg.norm(reference),1e-12)
    qc = {"active_factor_count":len(correlations),
          "minimum_factor_correlation":float(np.min(correlations)),
          "median_factor_correlation":float(np.median(correlations)),
          "normalized_rmse":float(normalized_rmse)}
    if qc["minimum_factor_correlation"] < 0.95 or qc["normalized_rmse"] > 0.15:
        raise RuntimeError(f"MOFA held-out projection QC failed: {qc}")
    return qc


def fit_mofa(processed, train_idx, test_idx, outer_fold, model_path,
             config_module):
    from mofapy2.run.entry_point import entry_point

    train_views = [processed[f"input_{view}"][train_idx].numpy().astype(float) for view in MOFA_VIEWS]
    test_views = [processed[f"input_{view}"][test_idx].numpy().astype(float) for view in MOFA_VIEWS]
    manifest = processed["participant_manifest"]
    features = [processed[f"{view}_features"] for view in MOFA_VIEWS]
    model = entry_point()
    model.set_data_options(scale_views=False,scale_groups=False,center_groups=True,use_float32=False)
    model.set_data_matrix(
        data=[[matrix] for matrix in train_views],
        likelihoods=["gaussian"]*3,views_names=list(MOFA_VIEWS),groups_names=["single_group"],
        samples_names=[manifest.iloc[train_idx]["id"].astype(str).tolist()],
        features_names=[[f"{view}::{name}" for name in names] for view,names in zip(MOFA_VIEWS,features)],
    )
    model.set_model_options(factors=MOFA_FACTORS,spikeslab_factors=False,
        spikeslab_weights=True,ard_factors=False,ard_weights=True)
    model.set_train_options(iter=1000,convergence_mode="medium",
        seed=config_module.RANDOM_STATE+1000*outer_fold,gpu_mode=False,
        verbose=False,quiet=True,dropR2=None)
    model.build(); model.run()
    z_train=np.asarray(model.model.nodes["Z"].getExpectations()["E"],dtype=float)
    weight_expectations=model.model.nodes["W"].getExpectations()
    weights=[np.asarray(x["E"],dtype=float) for x in weight_expectations]
    second=[np.asarray(x["E2"],dtype=float) for x in weight_expectations]
    tau=[collapse_tau_expectation(x["E"],matrix.shape[1]) for x,matrix in zip(
        model.model.nodes["Tau"].getExpectations(),train_views)]
    processed_train=[np.asarray(x,dtype=float) for x in model.data]
    intercepts=[np.asarray(model.intercepts[index][0],dtype=float) for index in range(3)]
    processed_test=[matrix-intercept[None,:] for matrix,intercept in zip(test_views,intercepts)]
    projected_train=project_gaussian_mofa(processed_train,weights,second,tau)
    qc=factor_projection_qc(z_train,projected_train)
    z_test=project_gaussian_mofa(processed_test,weights,second,tau)
    model_path.parent.mkdir(parents=True,exist_ok=True)
    model.save(outfile=str(model_path),save_data=False,expectations=["Z","W","Tau"])
    return z_train,z_test,{"mofapy2_version":version("mofapy2"),
        "initial_factors":MOFA_FACTORS,"retained_factors":z_train.shape[1],**qc}


def run_outer_fold(processed, model_name, outer_fold, config_module, output_root,
                   n_jobs=8):
    if model_name not in FINAL_MODELS:
        raise ValueError(f"Unreported comparator: {model_name}")
    started=time.time(); output_dir=Path(output_root)/model_name; output_dir.mkdir(parents=True,exist_ok=True)
    manifest=processed["participant_manifest"]; labels=manifest["benchmark_cluster"].to_numpy(); folds=manifest["outer_fold"].to_numpy()
    train_idx=np.where(folds!=outer_fold)[0]; test_idx=np.where(folds==outer_fold)[0]
    y_train,y_test=labels[train_idx],labels[test_idx]; matrix=multi_omics_matrix(processed)
    metadata={}; search=None
    if model_name=="pca_kmeans":
        estimator=PCA(n_components=0.95,svd_solver="full")
        z_train=estimator.fit_transform(matrix[train_idx]); z_test=estimator.transform(matrix[test_idx])
        params={"pca_variance_retained":0.95,"pca_n_components":int(estimator.n_components_),"kmeans_n_init":100}
    elif model_name=="mofa_kmeans":
        z_train,z_test,metadata=fit_mofa(processed,train_idx,test_idx,outer_fold,
            output_dir/"checkpoints"/f"outer_{outer_fold:02d}_mofa.hdf5",config_module)
        params={"initial_factors":16,"retained_factors":metadata["retained_factors"],
                "likelihoods":["gaussian"]*3,"spikeslab_weights":True,"ard_weights":True,"kmeans_n_init":100}
    else:
        params,search=elastic_inner_search(matrix,labels,train_idx,outer_fold,config_module,n_jobs)
        estimator=_classifier(params,config_module.RANDOM_STATE+1000*outer_fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",ConvergenceWarning); estimator.fit(matrix[train_idx],y_train)
        raw=estimator.predict(matrix[test_idx]).astype(int); mapped=raw.copy()

    if model_name.endswith("kmeans"):
        kmeans=KMeans(n_clusters=4,n_init=100,random_state=config_module.RANDOM_STATE+outer_fold)
        raw_train=kmeans.fit_predict(z_train); raw=kmeans.predict(z_test)
        mapping=derive_cluster_mapping(y_train,raw_train,CLUSTER_LABELS)
        mapped=apply_cluster_mapping(raw,mapping)
    overall,class_rows=calculate_metrics(y_test,raw,mapped,CLUSTER_LABELS)
    row={"model":model_name,"outer_fold":outer_fold,"train_n":len(train_idx),"test_n":len(test_idx),
         "best_params":json.dumps(params,sort_keys=True),"elapsed_minutes":(time.time()-started)/60,**metadata,**overall}
    pd.DataFrame([row]).to_csv(output_dir/f"outer_{outer_fold:02d}_metrics.csv",index=False)
    if search is not None: search.to_csv(output_dir/f"outer_{outer_fold:02d}_inner_search.csv",index=False)
    pd.DataFrame(class_rows).assign(model=model_name,outer_fold=outer_fold).to_csv(
        output_dir/f"outer_{outer_fold:02d}_class_metrics.csv",index=False)
    pd.DataFrame({"row_index":test_idx,"id":manifest.iloc[test_idx]["id"].to_numpy(),
        "model":model_name,"outer_fold":outer_fold,"true_cluster":y_test,
        "raw_predicted_cluster":raw,"mapped_predicted_cluster":mapped}).to_csv(
        output_dir/f"outer_{outer_fold:02d}_predictions.csv",index=False)
    return row


def aggregate(output_root, model_name):
    directory=Path(output_root)/model_name
    metrics=pd.concat([pd.read_csv(x) for x in sorted(directory.glob("outer_[0-9][0-9]_metrics.csv"))],ignore_index=True)
    classes=pd.concat([pd.read_csv(x) for x in sorted(directory.glob("outer_[0-9][0-9]_class_metrics.csv"))],ignore_index=True)
    predictions=pd.concat([pd.read_csv(x) for x in sorted(directory.glob("outer_[0-9][0-9]_predictions.csv"))],ignore_index=True).sort_values("row_index")
    metrics.to_csv(directory/"outer_fold_metrics.csv",index=False); classes.to_csv(directory/"class_metrics.csv",index=False); predictions.to_csv(directory/"participant_predictions.csv",index=False)
    return metrics,classes,predictions
