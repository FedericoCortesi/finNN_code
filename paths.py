from pathlib import Path

# Root of the repository = this file's parent directory
REPO_ROOT = Path(__file__).resolve().parent

# Common subpaths
MODELS_DIR = REPO_ROOT / "training_price_prediction" / "models"
DATA_DIR = REPO_ROOT / "data"
PIPELINE_DIR = REPO_ROOT / "pipeline"

# Example specific datasets
SP500_PATH = DATA_DIR / "sp500_daily_data.parquet"
PERMNOS_PATH = DATA_DIR / "permnos_list.txt"
INFO_PATH = DATA_DIR / "permnos_info.csv"
