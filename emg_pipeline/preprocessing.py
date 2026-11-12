"""Signal preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from .config import (
    DEFAULT_FILTER_ORDER,
    DEFAULT_HIGH_CUT,
    DEFAULT_LOW_CUT,
    DEFAULT_SAMPLE_RATE,
)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = DEFAULT_LOW_CUT,
    highcut: float = DEFAULT_HIGH_CUT,
    fs: float = DEFAULT_SAMPLE_RATE,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Band-pass filter EMG channels along the time axis."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, data, axis=-1)


def window_signal(raw_emg: np.ndarray, window_len: int, overlap: int) -> list[np.ndarray]:
    """Slice an EMG matrix of shape (channels, timesteps) into sliding windows."""
    if window_len <= 0:
        raise ValueError("window_len must be positive")
    if overlap < 0 or overlap >= window_len:
        raise ValueError("overlap must be in [0, window_len)")

    timesteps = raw_emg.shape[1]
    step = window_len - overlap
    windows = []

    for start in range(0, timesteps - window_len + 1, step):
        end = start + window_len
        windows.append(raw_emg[:, start:end])

    return windows


def compute_channel_standardization(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel normalization statistics across windows and time."""
    mean = windows.mean(axis=(0, 2), keepdims=True)
    std = windows.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def standardize_windows(
    windows: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize windows using per-channel statistics."""
    if mean is None or std is None:
        mean, std = compute_channel_standardization(windows)
    return (windows - mean) / std, mean, std
