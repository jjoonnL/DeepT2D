import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from collections import Counter
import pandas as pd

def set_seed(seed):
    """Sets the seed for all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def cluster_accuracy(true_labels, predicted_labels):
    """Calculates clustering accuracy using the Hungarian algorithm."""
    cost_matrix = np.zeros((len(set(true_labels)), len(set(predicted_labels))))
    unique_true = sorted(list(set(true_labels)))
    unique_pred = sorted(list(set(predicted_labels)))
    
    for i, true_label in enumerate(unique_true):
        for j, predicted_label in enumerate(unique_pred):
            cost_matrix[i, j] = -np.sum((true_labels == true_label) & (predicted_labels == predicted_label))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    accuracy = -cost_matrix[row_ind, col_ind].sum() / len(true_labels)
    return accuracy

def normalize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def train_epoch(model, train_loader, optimizer, device):
    """Trains the model for a single epoch."""
    model.train()
    epoch_loss = 0.0
    for x1, x2, x3, y in train_loader:
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)

        optimizer.zero_grad()
        pred_clinical = model(x1, x2, x3)
        loss = F.mse_loss(pred_clinical, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def train_model(model, clinical_df, train_loader, optimizer, scheduler, device, epochs):
    """Manages the complete model training process."""
    print("Training started...")
    acc_list = []
    loss_list = []
    
    true_labels, _ = pd.factorize(clinical_df["kmeans_cluster"])

    epoch_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_bar:
        epoch_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        loss_list.append(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 50 == 0:
            pred_labels = model.k_means_clustering(train_loader, device=device)
            accuracy = cluster_accuracy(true_labels, pred_labels)
            acc_list.append(accuracy)
            
            epoch_bar.write(
                f"Epoch {epoch + 1:03d}: | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | LR: {current_lr:.6f}"
            )
            print("Cluster distribution:", Counter(pred_labels))

    print("Training finished.")
    
    final_labels = model.k_means_clustering(train_loader, device=device)
    clinical_df["mic_cluster"] = final_labels
    
    return acc_list, loss_list, clinical_df

def map_clusters_to_subtypes(clinical_df, cluster_col_name='mic_cluster'):
    """
    Maps numeric cluster IDs to diabetes subtypes (SIRD, SIDD, MOD, MARD)
    based on the mean values of key clinical features.
    """
    features = ["HOMA_IR", "hba1c", "bmi", "age_at_diagnosis"]
    subtypes = ["SIRD", "SIDD", "MOD", "MARD"]
    
    mapping = {}
    mean_values = clinical_df.groupby(cluster_col_name)[features].mean()
    
    assigned_clusters = set()
    for feature, subtype in zip(features, subtypes):
        # Find the cluster with the absolute highest value for the feature
        max_cluster = mean_values[feature].idxmax()
        
        # If this cluster has already been assigned to another subtype, fail the mapping
        if max_cluster in assigned_clusters:
            return clinical_df, None # Return None for the mapping
            
        mapping[int(max_cluster)] = subtype
        assigned_clusters.add(int(max_cluster))
        
    # Check if all clusters were uniquely mapped
    if len(assigned_clusters) < len(subtypes):
        return clinical_df, None

    clinical_df["mapped_cluster"] = clinical_df[cluster_col_name].map(mapping)
    return clinical_df, mapping
