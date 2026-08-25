import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import config
from data import load_input_data, prepare_dataset


def test_sample_preprocessing():
    raw_data = load_input_data(
        config.GENOTYPE_PATH,
        config.PROTEOME_PATH,
        config.METABOLITE_PATH,
        config.CLINICAL_PATH,
    )
    processed = prepare_dataset(
        raw_data=raw_data,
        clinical_targets=config.CLINICAL_TARGETS,
        genotype_metadata_columns=config.GENOTYPE_METADATA_COLUMNS,
        id_columns={
            "genotype": config.GENOTYPE_ID_COLUMN,
            "proteome": config.OMICS_ID_COLUMN,
            "metabolite": config.OMICS_ID_COLUMN,
            "clinical": config.CLINICAL_ID_COLUMN,
        },
        sex_column=config.SEX_COLUMN,
        n_clusters=config.NUM_CLUSTERS,
        random_state=config.RANDOM_STATE,
        outer_splits=config.OUTER_SPLITS,
        outer_random_state=config.OUTER_RANDOM_STATE,
    )

    assert processed["input_genotype"].shape == (1000, 200)
    assert processed["input_proteome"].shape == (1000, 50)
    assert processed["input_metabolite"].shape == (1000, 50)
    assert processed["output_clinical"].shape == (1000, 5)

    manifest = processed["participant_manifest"]
    assert manifest["id"].is_unique
    assert sorted(manifest["outer_fold"].unique()) == list(range(1, 11))
    assert manifest.groupby("outer_fold").size().sum() == 1000

    sex = manifest["sex"].to_numpy()
    for matrix_name in [
        "input_genotype",
        "input_proteome",
        "input_metabolite",
        "output_clinical",
    ]:
        matrix = processed[matrix_name].numpy()
        assert np.isfinite(matrix).all()
        for sex_value in np.unique(sex):
            sex_matrix = matrix[sex == sex_value]
            assert np.max(np.abs(sex_matrix.mean(axis=0))) < 1e-5
