# Bridging omics to phenotypes

Code accompanying **“Bridging omics to phenotypes: a deep learning framework to characterize multi-omics patterns associated with type 2 diabetes subtypes.”**

The repository contains the model, validation, comparator, and feature-attribution workflows used in the revised analysis. Synthetic data are included to demonstrate the required input format and to verify code execution. They do not reproduce the study results and must not be used for clinical interpretation.

## Repository structure

```text
DeepT2D/
├── data/
│   ├── sample_data/          synthetic input data
│   ├── raw/                  user-provided data (not tracked)
│   └── processed/            generated tensors and manifests (not tracked)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_hyperparameter_tuning.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_nested_cross_validation.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_evaluation_and_labeling.ipynb
│   ├── 07_feature_attribution.ipynb
│   └── 08_clinical_association.ipynb
├── src/                      reusable analysis code
├── tests/                    preprocessing and model checks
├── outputs/                  generated results (not tracked)
└── requirements.txt
```

## Installation

Python 3.12 was used for the revision analysis. Install PyTorch for the relevant CPU/CUDA platform first, then install the remaining packages.

```bash
git clone https://github.com/jjoonnL/DeepT2D.git
cd DeepT2D
conda create -n deept2d python=3.12
conda activate deept2d
pip install -r requirements.txt
```

MOFA requires `mofapy2==0.7.4`. On systems where `h5py` cannot be built with the system compiler, installing compatible NumPy and h5py binaries through conda before `pip install -r requirements.txt` is recommended.

## Input data

Four participant-level tables are required:

- genotype: one ID column and variant columns;
- proteome: one ID column and protein columns;
- metabolite: one ID column and metabolite columns;
- clinical: ID, sex, BMI, HbA1c, age at diagnosis, HOMA2-B, and HOMA2-IR.

IDs must be unique within each table. Notebook 01 retains participants present in all four tables, removes genotype features with zero variance in either sex, applies sex-specific cohort-level scaling, generates the clinical benchmark clusters, and saves fixed outer-fold assignments. Input paths and column names are defined in `src/config.py`.

## Analysis order

1. `01_data_preprocessing.ipynb` aligns and preprocesses the four data sources.
2. `02_hyperparameter_tuning.ipynb` provides a single-outer-fold tuning interface. The final tuning is nested inside Notebook 04.
3. `03_model_training.ipynb` trains the repeated full-cohort models used for stable labeling.
4. `04_nested_cross_validation.ipynb` runs the 10-fold outer/5-fold inner generalization analysis. Hyperparameters, training epochs, k-means centroids, and cluster mappings are determined from outer-training participants only.
5. `05_model_comparison.ipynb` evaluates single-omics models, PCA + k-means, MOFA with 16 factors + k-means, and Elastic Net using the same outer folds.
6. `06_evaluation_and_labeling.ipynb` evaluates the explicitly recorded full-cohort model runs and derives majority-vote subtype labels.
7. `07_feature_attribution.ipynb` performs full-cohort and held-out Integrated Gradients analyses. Positive feature sums are primary; absolute attribution rankings are retained as a sensitivity analysis.
8. `08_clinical_association.ipynb` contains the included-versus-excluded comparison and adjusted complication models. Private clinical input files are required.

Long-running cells are disabled by default. Set the corresponding `RUN_*` flag only after reviewing the paths and active environment.

## Validation design

The final generalization pipeline uses 10 outer folds and 5 inner folds. Inner folds select the reconstruction model hyperparameters and training epoch. The model is then refitted using all outer-training participants. K-means centroids and the cluster-to-subtype mapping are also fitted using outer-training participants, after which held-out participants are assigned to the fixed centroids.

All reported single-omics and comparator analyses reuse the same outer-fold assignments. Elastic Net is a supervised predictive benchmark; PCA and MOFA are unsupervised representation-learning comparators followed by k-means.

## Reproducibility and data privacy

Generated tensors, participant manifests, participant-level predictions, attribution matrices, checkpoints, and private clinical data are excluded from version control. The tracked sample data are synthetic. Manuscript figure-layout code is maintained separately from this analysis repository.

Run the local checks with:

```bash
pytest -q
```
