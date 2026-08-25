import numpy as np
import torch

from src import config
from src.evaluation import apply_cluster_mapping, derive_cluster_mapping
from src.models import build_mic
from src.training import train_fixed_epochs


def _small_data(n=128):
    generator = torch.Generator().manual_seed(214)
    data = {
        "genotype": torch.randn(n, 12, generator=generator),
        "proteome": torch.randn(n, 8, generator=generator),
        "metabolite": torch.randn(n, 6, generator=generator),
        "clinical": torch.randn(n, 5, generator=generator),
    }
    return data


def test_model_and_training_smoke():
    data = _small_data()
    input_dims = {name: data[name].shape[1] for name in (
        "genotype", "proteome", "metabolite"
    )}
    params = {"decoder_dim": [], "dropout": 0.2, "weight_decay": 1e-3}
    model, info = train_fixed_epochs(
        data, input_dims, params, 2, config, torch.device("cpu"), 214
    )
    prediction = model(data["genotype"], data["proteome"], data["metabolite"])
    assert prediction.shape == (128, 5)
    assert info["epochs"] == 2
    assert np.isfinite(info["final_train_loss"])


def test_mapping_uses_training_partition():
    training_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    training_clusters = np.array([2, 2, 0, 0, 3, 3, 1, 1])
    mapping = derive_cluster_mapping(
        training_labels, training_clusters, np.arange(4)
    )
    assert mapping == {2: 0, 0: 1, 3: 2, 1: 3}
    assert np.array_equal(
        apply_cluster_mapping(np.array([2, 0, 3, 1]), mapping),
        np.arange(4),
    )


def test_checkpoint_architecture():
    input_dims = {"genotype": 415, "proteome": 714, "metabolite": 294}
    params = {"decoder_dim": [64], "dropout": 0.2, "weight_decay": 1e-4}
    model = build_mic(input_dims, params, config)
    assert model.centroid.shape == (4, 16)
    assert model.genotype_encoder[0].weight.shape == (128, 415)
    assert model.proteome_encoder[0].weight.shape == (128, 714)
    assert model.metabolite_encoder[0].weight.shape == (128, 294)
    assert model.clinical_decoder[-1].weight.shape == (5, 64)
