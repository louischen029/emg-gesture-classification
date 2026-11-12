"""Shared configuration and label metadata."""

from pathlib import Path

GESTURE_MAP = {
    "noGesture": 0,
    "fist": 1,
    "open": 2,
    "pinch": 3,
    "waveIn": 4,
    "waveOut": 5,
}

GESTURE_NAMES = {value: key for key, value in GESTURE_MAP.items()}

DEFAULT_SAMPLE_RATE = 200.0
DEFAULT_LOW_CUT = 20.0
DEFAULT_HIGH_CUT = 95.0
DEFAULT_FILTER_ORDER = 4

DEFAULT_WINDOW_LEN = 100
DEFAULT_OVERLAP = 20

DEFAULT_TRAINING_PATH = Path("data") / "training" / "trainingJSON"
DEFAULT_TESTING_PATH = Path("data") / "testing" / "testingJSON"
