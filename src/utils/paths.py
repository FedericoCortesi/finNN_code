from pathlib import Path

# Root of the repository = this file's parent directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Common subpaths
SRC_DIR = REPO_ROOT / "src" 
EXPERIMENTS_DIR = REPO_ROOT / "src" / "price_prediction" / "experiments"
SMOKE_DIR = REPO_ROOT / "src" / "price_prediction" / "experiments" / "smoke"
DATA_DIR = REPO_ROOT / "src" / "data"
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"
CONFIG_DIR = REPO_ROOT / "src" / "config"


# Example specific datasets
SP500_PATH = DATA_DIR / "sp500_daily_data.parquet"
SP500COPY_PATH = DATA_DIR / "sp500_daily_data_copy.parquet" # for debugging
PERMNOS_PATH = DATA_DIR / "permnos_list.txt"
INFO_PATH = DATA_DIR / "permnos_info.csv"
