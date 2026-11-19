"""Runtime configuration for the detection API."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Config:
    MODEL_DIR = os.environ.get("RSW_MODEL_DIR", os.path.join(REPO_ROOT, "models"))
    SEQUENCE_LENGTH = 50
    MONITORING_WINDOW = 5.0  # 5 seconds
    DETECTION_THRESHOLD = 0.5
    MAX_PROCESSES = 100
    UPDATE_INTERVAL = 1.0
    BUFFER_SIZE = 1000


config = Config()
