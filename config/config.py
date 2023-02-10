from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")
RESULTS_DIR = Path(BASE_DIR, "results")
IMGCROP_DIR = Path(RESULTS_DIR, "img_crop")
RAWDATA_DIR = Path(RESULTS_DIR, "raw_data")
DATARESEARCH_DIR = Path(BASE_DIR, "data_research")

# Add to path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "strong_sort"))

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMGCROP_DIR.mkdir(parents=True, exist_ok=True)
RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "strong_sort").mkdir(parents=True, exist_ok=True)


TABLES_CONFIG = {
    "user_embeddings": ["id_user", "embedding"],
    "info_users": ["id_user", "name", "age", "phone", "accept"]
}

EMBEDDING_DIMENSION = 512