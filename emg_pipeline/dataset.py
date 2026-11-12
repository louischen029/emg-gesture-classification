"""Dataset loading and JSON deserialization for the EMG corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DEFAULT_OVERLAP, DEFAULT_TRAINING_PATH, DEFAULT_WINDOW_LEN, GESTURE_MAP
from .preprocessing import butter_bandpass_filter, window_signal


@dataclass
class DatasetSummary:
    users: int
    trials: int
    windows: int
    labels: dict[str, int]
    user_names: list[str]


@dataclass
class WindowMetadata:
    group_id: int
    user_name: str
    trial_key: str
    gesture_name: str


@dataclass
class SequenceMetadata:
    group_id: int
    user_name: str
    trial_key: str
    gesture_name: str


def discover_user_json_files(
    parent_path: str | Path,
    max_users: int | None = None,
    sample_users: int | None = None,
    random_seed: int = 42,
) -> list[Path]:
    """Return user JSON files with optional user-level subsampling."""
    root = Path(parent_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    user_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")],
        key=lambda path: int(path.name.replace("user", "")) if path.name.replace("user", "").isdigit() else path.name,
    )
    if max_users is not None:
        user_dirs = user_dirs[:max_users]

    if sample_users is not None:
        if sample_users <= 0:
            raise ValueError("sample_users must be positive")
        if sample_users < len(user_dirs):
            rng = np.random.default_rng(random_seed)
            chosen_indices = np.sort(rng.choice(len(user_dirs), size=sample_users, replace=False))
            user_dirs = [user_dirs[index] for index in chosen_indices]

    json_files = []
    for user_dir in user_dirs:
        json_files.extend(sorted(user_dir.glob("*.json")))
    return json_files


def iter_training_samples(
    parent_path: str | Path,
    max_users: int | None = None,
    sample_users: int | None = None,
    random_seed: int = 42,
) -> Iterator[tuple[int, Path, str, dict]]:
    """Yield `(group_id, json_file, sample_key, sample_data)` for every training sample."""
    json_files = discover_user_json_files(
        parent_path,
        max_users=max_users,
        sample_users=sample_users,
        random_seed=random_seed,
    )
    for group_id, json_path in enumerate(json_files):
        try:
            payload = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue

        training_samples = payload.get("trainingSamples", {})
        if not isinstance(training_samples, dict):
            continue

        for sample_key, sample_data in training_samples.items():
            if isinstance(sample_data, dict):
                yield group_id, json_path, sample_key, sample_data


def build_emg_matrix(emg_dict: dict) -> np.ndarray | None:
    """Reconstruct a stable `(8, timesteps)` float matrix from channel dictionaries."""
    if not isinstance(emg_dict, dict):
        return None

    try:
        channels = [emg_dict[f"ch{i}"] for i in range(1, 9)]
    except KeyError:
        return None

    lengths = [len(channel) for channel in channels]
    if not lengths or min(lengths) == 0:
        return None

    min_length = min(lengths)
    clipped = [channel[:min_length] for channel in channels]
    raw_emg = np.asarray(clipped, dtype=np.float32)

    if raw_emg.ndim != 2 or raw_emg.shape[0] != 8:
        return None

    return raw_emg


def rms_sequence_from_emg(
    raw_emg: np.ndarray,
    window_len: int = 5,
    stride: int = 5,
) -> np.ndarray | None:
    """Convert one `(8, timesteps)` EMG trial into a sequence of RMS vectors."""
    timesteps = raw_emg.shape[1]
    if timesteps < window_len:
        return None

    vectors = []
    for start in range(0, timesteps - window_len + 1, stride):
        window = raw_emg[:, start : start + window_len]
        vectors.append(np.sqrt(np.mean(window**2, axis=1)))

    if not vectors:
        return None

    return np.asarray(vectors, dtype=np.float32)


def load_window_dataset_with_metadata(
    parent_path: str | Path = DEFAULT_TRAINING_PATH,
    window_len: int = DEFAULT_WINDOW_LEN,
    overlap: int = DEFAULT_OVERLAP,
    apply_filter: bool = True,
    max_users: int | None = None,
    sample_users: int | None = None,
    max_windows_per_user: int | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[WindowMetadata], DatasetSummary]:
    """Load all users into a windowed dataset aligned with labels, groups, and trial metadata."""
    windows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    metadata: list[WindowMetadata] = []

    label_counts = {label: 0 for label in GESTURE_MAP}
    trial_count = 0
    users_seen: set[int] = set()
    user_names: dict[int, str] = {}
    windows_per_user: dict[int, int] = {}

    for group_id, json_path, sample_key, sample_data in iter_training_samples(
        parent_path,
        max_users=max_users,
        sample_users=sample_users,
        random_seed=random_seed,
    ):
        gesture_name = sample_data.get("gestureName")
        emg_matrix = build_emg_matrix(sample_data.get("emg"))
        if gesture_name not in GESTURE_MAP or emg_matrix is None:
            continue

        if apply_filter:
            emg_matrix = butter_bandpass_filter(emg_matrix)

        sample_windows = window_signal(emg_matrix, window_len=window_len, overlap=overlap)
        if not sample_windows:
            continue

        if max_windows_per_user is not None:
            used = windows_per_user.get(group_id, 0)
            remaining = max_windows_per_user - used
            if remaining <= 0:
                continue
            sample_windows = sample_windows[:remaining]
            if not sample_windows:
                continue

        trial_count += 1
        users_seen.add(group_id)
        user_names[group_id] = json_path.stem
        windows_per_user[group_id] = windows_per_user.get(group_id, 0) + len(sample_windows)
        label_counts[gesture_name] += len(sample_windows)

        numeric_label = GESTURE_MAP[gesture_name]
        windows.extend(sample_windows)
        labels.extend([numeric_label] * len(sample_windows))
        groups.extend([group_id] * len(sample_windows))
        metadata.extend(
            [
                WindowMetadata(
                    group_id=group_id,
                    user_name=json_path.stem,
                    trial_key=sample_key,
                    gesture_name=gesture_name,
                )
                for _ in sample_windows
            ]
        )

    if not windows:
        empty = np.array([])
        summary = DatasetSummary(users=0, trials=0, windows=0, labels=label_counts, user_names=[])
        return empty, empty, empty, [], summary

    X = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype=np.int64)
    summary = DatasetSummary(
        users=len(users_seen),
        trials=trial_count,
        windows=len(windows),
        labels=label_counts,
        user_names=[user_names[group_id] for group_id in sorted(user_names)],
    )
    return X, y, group_array, metadata, summary


def load_window_dataset(
    parent_path: str | Path = DEFAULT_TRAINING_PATH,
    window_len: int = DEFAULT_WINDOW_LEN,
    overlap: int = DEFAULT_OVERLAP,
    apply_filter: bool = True,
    max_users: int | None = None,
    sample_users: int | None = None,
    max_windows_per_user: int | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DatasetSummary]:
    """Load all users into a windowed dataset aligned with labels and user groups."""
    X, y, group_array, _, summary = load_window_dataset_with_metadata(
        parent_path=parent_path,
        window_len=window_len,
        overlap=overlap,
        apply_filter=apply_filter,
        max_users=max_users,
        sample_users=sample_users,
        max_windows_per_user=max_windows_per_user,
        random_seed=random_seed,
    )
    return X, y, group_array, summary


def load_rms_sequence_dataset(
    parent_path: str | Path = DEFAULT_TRAINING_PATH,
    rms_window_len: int = 5,
    rms_stride: int = 5,
    apply_filter: bool = False,
    max_users: int | None = None,
    sample_users: int | None = None,
    max_trials_per_user: int | None = None,
    max_trials_per_gesture_per_user: int | None = None,
    random_seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[SequenceMetadata], DatasetSummary]:
    """Load EMG trials as variable-length RMS sequences for temporal models."""
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    metadata: list[SequenceMetadata] = []

    label_counts = {label: 0 for label in GESTURE_MAP}
    trial_count = 0
    users_seen: set[int] = set()
    user_names: dict[int, str] = {}
    trials_per_user: dict[int, int] = {}
    trials_per_user_gesture: dict[tuple[int, str], int] = {}

    for group_id, json_path, sample_key, sample_data in iter_training_samples(
        parent_path,
        max_users=max_users,
        sample_users=sample_users,
        random_seed=random_seed,
    ):
        gesture_name = sample_data.get("gestureName")
        emg_matrix = build_emg_matrix(sample_data.get("emg"))
        if gesture_name not in GESTURE_MAP or emg_matrix is None:
            continue

        if max_trials_per_user is not None and trials_per_user.get(group_id, 0) >= max_trials_per_user:
            continue
        if (
            max_trials_per_gesture_per_user is not None
            and trials_per_user_gesture.get((group_id, gesture_name), 0) >= max_trials_per_gesture_per_user
        ):
            continue

        if apply_filter:
            emg_matrix = butter_bandpass_filter(emg_matrix)

        rms_sequence = rms_sequence_from_emg(
            emg_matrix,
            window_len=rms_window_len,
            stride=rms_stride,
        )
        if rms_sequence is None or len(rms_sequence) == 0:
            continue

        trial_count += 1
        users_seen.add(group_id)
        user_names[group_id] = json_path.stem
        trials_per_user[group_id] = trials_per_user.get(group_id, 0) + 1
        trials_per_user_gesture[(group_id, gesture_name)] = trials_per_user_gesture.get((group_id, gesture_name), 0) + 1
        label_counts[gesture_name] += 1

        sequences.append(rms_sequence)
        labels.append(GESTURE_MAP[gesture_name])
        groups.append(group_id)
        metadata.append(
            SequenceMetadata(
                group_id=group_id,
                user_name=json_path.stem,
                trial_key=sample_key,
                gesture_name=gesture_name,
            )
        )

    if not sequences:
        empty = np.array([])
        summary = DatasetSummary(users=0, trials=0, windows=0, labels=label_counts, user_names=[])
        return [], empty, empty, [], summary

    summary = DatasetSummary(
        users=len(users_seen),
        trials=trial_count,
        windows=sum(len(sequence) for sequence in sequences),
        labels=label_counts,
        user_names=[user_names[group_id] for group_id in sorted(user_names)],
    )
    return sequences, np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64), metadata, summary
