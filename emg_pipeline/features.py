"""Handcrafted time-domain feature extraction."""

from __future__ import annotations

import numpy as np


def _safe_log_detector(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Geometric-mean-style log detector used in EMG literature."""
    return np.exp(np.mean(np.log(np.abs(values) + eps), axis=1))


def _normalize_window(window: np.ndarray, method: str = "channel_zscore", eps: float = 1e-6) -> np.ndarray:
    """Normalize a single `(channels, timesteps)` window for subject-robust features."""
    if method == "channel_zscore":
        mean = np.mean(window, axis=1, keepdims=True)
        std = np.std(window, axis=1, keepdims=True)
        std[std < eps] = 1.0
        return (window - mean) / std

    if method == "channel_rms":
        rms = np.sqrt(np.mean(window**2, axis=1, keepdims=True))
        rms[rms < eps] = 1.0
        return window / rms

    raise ValueError(f"Unsupported normalization method: {method}")


def _channel_correlation_features(window: np.ndarray) -> np.ndarray:
    """Upper-triangular inter-channel correlation features."""
    correlation = np.corrcoef(window)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    upper_indices = np.triu_indices(correlation.shape[0], k=1)
    return correlation[upper_indices].astype(np.float32)


def _frequency_features(
    window: np.ndarray,
    sample_rate: float = 200.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute compact spectral descriptors per channel."""
    spectrum = np.abs(np.fft.rfft(window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(window.shape[1], d=1.0 / sample_rate)

    total_power = np.sum(spectrum, axis=1) + eps
    mean_frequency = np.sum(spectrum * freqs[None, :], axis=1) / total_power

    cumulative_power = np.cumsum(spectrum, axis=1)
    median_indices = np.argmax(cumulative_power >= (total_power[:, None] / 2.0), axis=1)
    median_frequency = freqs[median_indices]

    power_distribution = spectrum / total_power[:, None]
    spectral_entropy = -np.sum(power_distribution * np.log(power_distribution + eps), axis=1)

    return np.concatenate(
        [total_power, mean_frequency, median_frequency, spectral_entropy]
    ).astype(np.float32)


def extract_time_domain_features(
    window: np.ndarray,
    zc_threshold: float = 0.01,
    ssc_threshold: float = 0.0,
) -> np.ndarray:
    """Compute the original 40 time-domain features from one `(8, window_len)` EMG window."""
    mav = np.mean(np.abs(window), axis=1)
    rms = np.sqrt(np.mean(window**2, axis=1))
    wl = np.sum(np.abs(np.diff(window, axis=1)), axis=1)

    left = window[:, :-1]
    right = window[:, 1:]
    zc = np.sum(
        (left * right < 0)
        & (np.abs(left - right) >= zc_threshold),
        axis=1,
    )

    diffs = np.diff(window, axis=1)
    ssc = np.sum(
        (diffs[:, :-1] * diffs[:, 1:] < 0)
        & (np.abs(diffs[:, :-1] - diffs[:, 1:]) >= ssc_threshold),
        axis=1,
    )

    return np.concatenate([mav, rms, wl, zc, ssc]).astype(np.float32)


def extract_extended_time_domain_features(
    window: np.ndarray,
    zc_threshold: float = 0.01,
    ssc_threshold: float = 0.0,
    wamp_threshold: float = 0.02,
) -> np.ndarray:
    """Compute an expanded EMG feature vector with common time-domain descriptors."""
    base_features = extract_time_domain_features(
        window,
        zc_threshold=zc_threshold,
        ssc_threshold=ssc_threshold,
    )

    variance = np.var(window, axis=1)
    iemg = np.sum(np.abs(window), axis=1)
    aac = np.mean(np.abs(np.diff(window, axis=1)), axis=1)
    log_detector = _safe_log_detector(window)
    dasdv = np.sqrt(np.mean(np.diff(window, axis=1) ** 2, axis=1))
    wamp = np.sum(np.abs(np.diff(window, axis=1)) >= wamp_threshold, axis=1)

    return np.concatenate(
        [base_features, variance, iemg, aac, log_detector, dasdv, wamp]
    ).astype(np.float32)


def extract_hybrid_features(
    window: np.ndarray,
    zc_threshold: float = 0.01,
    ssc_threshold: float = 0.0,
    wamp_threshold: float = 0.02,
    sample_rate: float = 200.0,
) -> np.ndarray:
    """Combine extended time-domain and compact frequency-domain features."""
    extended = extract_extended_time_domain_features(
        window,
        zc_threshold=zc_threshold,
        ssc_threshold=ssc_threshold,
        wamp_threshold=wamp_threshold,
    )
    spectral = _frequency_features(window, sample_rate=sample_rate)
    return np.concatenate([extended, spectral]).astype(np.float32)


def extract_subject_robust_features(
    window: np.ndarray,
    sample_rate: float = 200.0,
) -> np.ndarray:
    """Combine raw, normalized, and inter-channel structure features."""
    raw_hybrid = extract_hybrid_features(window, sample_rate=sample_rate)
    zscore_hybrid = extract_hybrid_features(
        _normalize_window(window, method="channel_zscore"),
        sample_rate=sample_rate,
    )
    rms_hybrid = extract_hybrid_features(
        _normalize_window(window, method="channel_rms"),
        sample_rate=sample_rate,
    )
    channel_corr = _channel_correlation_features(window)
    return np.concatenate([raw_hybrid, zscore_hybrid, rms_hybrid, channel_corr]).astype(np.float32)


def extract_feature_dataset(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.0,
    feature_set: str = "base",
    sample_rate: float = 200.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert window tensors into a tabular feature matrix."""
    feature_rows = []
    clean_labels = []
    clean_groups = []

    for index, window in enumerate(windows):
        mav = np.mean(np.abs(window), axis=1)
        avg_window_mav = float(np.mean(mav))

        if labels[index] != 0 and avg_window_mav < threshold:
            continue

        if feature_set == "base":
            feature_row = extract_time_domain_features(window)
        elif feature_set == "extended":
            feature_row = extract_extended_time_domain_features(window)
        elif feature_set == "hybrid":
            feature_row = extract_hybrid_features(window, sample_rate=sample_rate)
        elif feature_set == "subject_robust":
            feature_row = extract_subject_robust_features(window, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unsupported feature_set: {feature_set}")

        feature_rows.append(feature_row)
        clean_labels.append(labels[index])
        clean_groups.append(groups[index])

    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(clean_labels, dtype=np.int64),
        np.asarray(clean_groups, dtype=np.int64),
    )
