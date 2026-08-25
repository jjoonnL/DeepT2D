import numpy as np
import pandas as pd
import torch

from src.generalization import input_dimensions, model_data, parameter_key


def test_data_subsetting_and_parameter_key():
    processed = {
        "input_genotype": torch.arange(60).reshape(10, 6).float(),
        "input_proteome": torch.arange(40).reshape(10, 4).float(),
        "input_metabolite": torch.arange(30).reshape(10, 3).float(),
        "output_clinical": torch.arange(50).reshape(10, 5).float(),
        "participant_manifest": pd.DataFrame({
            "benchmark_cluster": np.arange(10) % 4,
            "outer_fold": np.arange(10) + 1,
        }),
    }
    subset = model_data(processed, np.array([1, 4, 7]))
    assert subset["genotype"].shape == (3, 6)
    assert input_dimensions(processed) == {
        "genotype": 6, "proteome": 4, "metabolite": 3
    }
    assert parameter_key({
        "weight_decay": 0.001, "dropout": 0.2, "decoder_dim": []
    }) == '{"decoder_dim": [], "dropout": 0.2, "weight_decay": 0.001}'
