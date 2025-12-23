# Bridging omics to phenotypes: a deep learning framework to characterize the molecular basis of type 2 diabetes subtypes

This repository contains the official source code and sample data for the paper "Bridging omics to phenotypes: a deep learning framework to characterize the molecular basis of type 2 diabetes subtypes".

## Project Structure

```
T2D_subtype_analysis/
├── README.md                 <- The file you are reading
├── notebooks/                <- Contains the analysis pipeline as Jupyter notebooks.
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_hyperparameter_tuning.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_nested_cross_validation.ipynb
│   ├── 05_evaluation_and_labeling.ipynb
│   ├── 06_feature_analysis.ipynb
├── src/                      <- Contains modularized Python source code.
│   ├── config.py             # Stores all configurations, paths, and hyperparameters.
│   ├── models.py             # Defines models.
│   └── utils.py              # Contains helper functions for training, evaluation, etc.
├── data/                     <- Directory for all data.
│   ├── sample_data/          # Contains anonymized sample data to run the pipeline.
│   └── processed/            # Processed, model-ready data will be saved here.
├── outputs/                  <- Directory for all outputs.
│   ├── models/               # Saved model weights (.pth) are stored here.
│   └── ...                   # Other outputs like results CSVs are saved here.
└── requirements.txt          <- Lists all Python package dependencies.
```

## Setup

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [[repository-url]](https://github.com/jjoonnL/T2D.git)
    cd T2D_subtype_analysis
    ```

2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda create -n t2d_analysis python=3.12
    conda activate t2d_analysis
    ```

3.  **Install the required packages:**

    **Note:** Please install [PyTorch](https://pytorch.org/get-started/locally/) according to your compute platform (CUDA version) before installing other dependencies.
    
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Analysis

The entire analysis pipeline is organized into Jupyter notebooks within the `notebooks/` directory. They are numbered and should be run sequentially.

### Using the Sample Data

This repository includes a small, anonymized sample dataset located in `data/sample_data/` to allow for immediate execution and verification of the pipeline.

By default, the configuration in `src/config.py` is set up to use this sample data. No changes are needed to run the pipeline out-of-the-box.

### Execution Order

1.  **`01_data_preprocessing.ipynb`**:
    *   **Action:** Loads, normalizes, and filters the sample data.
    *   **Output:** A single `processed_dataset.pt` file in `data/processed/`.

2.  **`02_hyperparameter_tuning.ipynb`**:
    *   **Action:** Performs a grid search with cross-validation to find the optimal model hyperparameters.
    *   **Output:** `hyperparameter_tuning_results.csv` in `outputs/`.

3.  **`03_model_training.ipynb`**:
    *   **Action:** Trains the model for 100 independent runs using the best hyperparameters found in `src/config.py`.
    *   **Output:** 100 trained model weight files (`.pth`) in `outputs/models/`.

4.  **`04_nested_cross_validation.ipynb`**:
    *   **Action:** Evaluates the model's generalization performance using a robust nested cross-validation scheme.
    *   **Output:** `nested_cv_results.csv` and benchmark comparison files in `outputs/`.

5.  **`05_evaluation_and_labeling.ipynb`**:
    *   **Action:** Evaluates all 100 trained models and assigns a final, stable subtype label to each sample via majority vote.
    *   **Output:** `final_clinical_data_with_labels.csv` in `data/processed/`.

6.  **`06_feature_analysis.ipynb`**:
    *   **Action:** Performs feature importance analysis using Integrated Gradients to identify key molecular drivers for each subtype.
    *   **Output:** Analysis results and visualizations within the notebook.

### Using Your Own Data

**Note:** Your input data files must follow the same format and column structure as the provided sample data (see `data/sample_data/`). Ensure that the Sample IDs are consistent across all omics and clinical files.
1.  Place your raw data files in a directory of your choice (e.g., `data/raw/`).
2.  Open `src/config.py`.
3.  Update the file paths in the `DATA PATHS` section to point to your data files.

    ```python
    # src/config.py

    # ...
    # Change this to your data directory
    RAW_DATA_DIR = DATA_DIR / "raw" 
    
    # Update these paths to point to your files
    GENOTYPE_PATH = RAW_DATA_DIR / "your_genotype_file.csv"
    PROTEOME_PATH = RAW_DATA_DIR / "your_proteome_file.csv"
    METABOLITE_PATH = RAW_DATA_DIR / "your_metabolite_file.csv"
    CLINICAL_PATH = RAW_DATA_DIR / "your_clinical_file.csv"
    # ...
    ```
4.  Run the notebooks sequentially as described above.
