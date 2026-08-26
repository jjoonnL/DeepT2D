from pathlib import Path

# ===================================================================
#  BASE PATH CONFIGURATION
# ===================================================================
# Set the project base directory relative to this file's location.
# This makes the project portable across different machines.
BASE_DIR = Path(__file__).resolve().parent.parent

# ===================================================================
#  DATA PATHS
# ===================================================================
# As per the README, raw data should be placed in the `data/raw` directory.
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Input data paths
# -------------------------------------------------------------------
# OPTION 1: USE SAMPLE DATA (DEFAULT)
# -------------------------------------------------------------------
# By default, the pipeline uses the anonymized sample data provided in this repository.
# These paths are relative to the project root.
GENOTYPE_PATH = DATA_DIR / "sample_data/genotype.csv"
PROTEOME_PATH = DATA_DIR / "sample_data/proteome.csv"
METABOLITE_PATH = DATA_DIR / "sample_data/metabolite.csv"
CLINICAL_PATH = DATA_DIR / "sample_data/clinical.csv"

# Input schema
GENOTYPE_ID_COLUMN = "IID"
OMICS_ID_COLUMN = "id"
CLINICAL_ID_COLUMN = "id"
SEX_COLUMN = "sex"

GENOTYPE_METADATA_COLUMNS = [
    "FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"
]

CLINICAL_TARGETS = [
    "bmi",
    "hba1c",
    "age_at_diagnosis",
    "HOMA_B",
    "HOMA_IR",
]

OUTER_SPLITS = 10
OUTER_RANDOM_STATE = 42
INNER_SPLITS = 5
INNER_RANDOM_STATE = 42

# -------------------------------------------------------------------
# OPTION 2: USE YOUR OWN DATA (CUSTOM)
# -------------------------------------------------------------------
# To use your own dataset, uncomment the lines below and update these paths to point to your files.
# GENOTYPE_PATH = RAW_DATA_DIR / "your_genotype_file.csv"
# PROTEOME_PATH = RAW_DATA_DIR / "your_proteome_file.csv"
# METABOLITE_PATH = RAW_DATA_DIR / "your_metabolite_file.csv"
# CLINICAL_PATH = RAW_DATA_DIR / "your_clinical_file.csv"

# ===================================================================
#  OUTPUT PATHS
# ===================================================================
# Configure directories to save model weights and figures.
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_SAVE_DIR = OUTPUT_DIR / "models"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_dataset.pt"
PARTICIPANT_MANIFEST_PATH = PROCESSED_DATA_DIR / "participant_manifest.csv"

# ===================================================================
#  MODEL HYPERPARAMETERS
# ===================================================================
# --- Model Architecture ---
# NOTE: Input dimensions are inferred automatically from the data in the notebooks.
# These lists define the hidden layers of each component.

ENCODER_DIMS = [128, 32]        # Hidden layers for all omics encoders
INTEGRATION_DIMS = [64]         # Hidden layers for the integration block
LATENT_DIM = 16                 # Dimension of the final latent space
CLINICAL_OUTPUT_DIM = 5         # Final output dimension (number of clinical variables)

# --- Hyperparameters used for full-cohort model training ---
DECODER_DIMS = []               # Hidden layers for the decoder. E.g., [], [32], or [64]
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4

# --- Training Parameters ---
EPOCHS = 50
MAX_EPOCHS = 300
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-6
LEARNING_RATE = 0.01
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.9
BATCH_SIZE = 64
RANDOM_STATE = 214
NUM_RUNS = 100 # Total training iterations with different random seeds
NUM_CLUSTERS = 4

PARAM_GRID = {
    "decoder_dim": [[], [32], [64]],
    "dropout": [0.1, 0.2, 0.3],
    "weight_decay": [1e-4, 1e-3, 1e-2],
}
