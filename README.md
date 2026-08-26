# Bridging omics to phenotypes: a deep learning framework to characterize multi-omics patterns associated with type 2 diabetes subtypes

This repository contains the source code and sample data for the paper “Bridging omics to phenotypes: a deep learning framework to characterize multi-omics patterns associated with type 2 diabetes subtypes.”

## Project Structure

```text
DeepT2D/
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_hyperparameter_tuning.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_nested_cross_validation.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_evaluation_and_labeling.ipynb
│   └── 07_feature_analysis.ipynb
├── src/                      model definitions and helper functions
├── data/
│   ├── sample_data/
│   └── processed/
├── outputs/
└── requirements.txt
```

## Setup

1. Clone the repository.

   ```bash
   git clone https://github.com/jjoonnL/DeepT2D.git
   cd DeepT2D
   ```

2. Create and activate a Conda environment.

   ```bash
   conda create -n deept2d python=3.12
   conda activate deept2d
   ```

3. Install [PyTorch](https://pytorch.org/get-started/locally/) for the relevant compute platform, followed by the remaining dependencies.

   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Analysis

The analysis is organized into numbered Jupyter notebooks in the `notebooks/` directory.

1. `01_data_preprocessing.ipynb` prepares the genotype, proteome, metabolite, and clinical data.
2. `02_hyperparameter_tuning.ipynb` performs model hyperparameter tuning.
3. `03_model_training.ipynb` trains the full-cohort models.
4. `04_nested_cross_validation.ipynb` evaluates held-out generalization performance.
5. `05_model_comparison.ipynb` evaluates the single-omics models and comparator methods.
6. `06_evaluation_and_labeling.ipynb` evaluates the full-cohort models and derives the final subtype labels.
7. `07_feature_analysis.ipynb` performs full-cohort and held-out Integrated Gradients analyses.

### Using the Sample Data

Synthetic sample data are provided in `data/sample_data/` to demonstrate the expected input format and verify that the pipeline runs. They do not reproduce the study results and should not be used for clinical interpretation.

### Using Your Own Data

Input files should follow the formats shown in `data/sample_data/`, with consistent participant IDs across all data sources. To use other data, update the input paths in `src/config.py` and run the notebooks in order.
