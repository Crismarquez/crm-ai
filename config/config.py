from pathlib import Path
import sys
import logging
import logging.config
from rich.logging import RichHandler

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")
RESULTS_DIR = Path(BASE_DIR, "results")
IMGCROP_DIR = Path(RESULTS_DIR, "img_crop")
RAWDATA_DIR = Path(RESULTS_DIR, "raw_data")
DATARESEARCH_DIR = Path(BASE_DIR, "data_research")
LOGS_DIR = Path(BASE_DIR, "logs")

# Add to path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "strong_sort"))

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMGCROP_DIR.mkdir(parents=True, exist_ok=True)
RAWDATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "strong_sort").mkdir(parents=True, exist_ok=True)


TABLES_CONFIG = {
    "user_embeddings": ["id_user", "embedding"],
    "info_users": ["id_user", "name", "age", "phone", "accept"],
}

EMBEDDING_DIMENSION = 512
DISTANCE_EYES_THRESHOLD = 80
N_EMBEDDINGS = 5
N_EMBEDDINGS_REGISTER = 5

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
