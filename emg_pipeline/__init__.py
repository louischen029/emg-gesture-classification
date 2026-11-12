"""Core package for the EMG gesture recognition project."""

from .config import DEFAULT_TRAINING_PATH, GESTURE_MAP, GESTURE_NAMES
from .dataset import DatasetSummary, load_window_dataset
from .features import extract_feature_dataset
from .models import EMG1DCNN, WindowTensorDataset

__all__ = [
    "DEFAULT_TRAINING_PATH",
    "DatasetSummary",
    "EMG1DCNN",
    "GESTURE_MAP",
    "GESTURE_NAMES",
    "WindowTensorDataset",
    "extract_feature_dataset",
    "load_window_dataset",
]
