"""Training and evaluation routines for EMG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .config import GESTURE_MAP
from .dataset import SequenceMetadata, WindowMetadata
from .features import extract_feature_dataset
from .models import EMG1DCNN, EMGSequenceLSTM, SequenceTensorDataset, WindowTensorDataset, collate_padded_sequences
from .preprocessing import standardize_windows


@dataclass
class FeatureExperimentResult:
    original_size: int
    cleaned_size: int
    feature_dim: int
    fold_accuracies: list[float]
    mean_accuracy: float
    report: str
    confusion: np.ndarray


@dataclass
class SweepRow:
    model_name: str
    feature_set: str
    energy_threshold: float
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None
    mean_accuracy: float
    std_accuracy: float
    cleaned_size: int
    feature_dim: int


@dataclass
class HeldoutEvaluationResult:
    train_size: int
    test_size: int
    feature_dim: int
    window_accuracy: float
    trial_accuracy: float
    report: str
    confusion: np.ndarray
    users_evaluated: int = 0


@dataclass
class HeldoutSweepRow:
    mode: str
    feature_set: str
    model_name: str
    energy_threshold: float
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    class_weight: str | None
    feature_dim: int
    window_accuracy: float
    trial_accuracy: float
    users_evaluated: int


@dataclass
class CNNExperimentResult:
    train_loss_history: list[float]
    val_accuracy_history: list[float]
    report: str


@dataclass
class SequenceHeldoutResult:
    train_sequences: int
    test_sequences: int
    window_size: int
    stride: int
    best_val_accuracy: float
    test_accuracy: float
    test_trial_accuracy: float
    report: str
    confusion: np.ndarray


def run_feature_baseline(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    energy_threshold: float = 0.0,
    n_splits: int = 5,
    n_estimators: int = 100,
    max_depth: int | None = 15,
    random_seed: int = 42,
    feature_set: str = "base",
    model_name: str = "random_forest",
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
    collect_diagnostics: bool = True,
    use_stratified_groups: bool = True,
) -> FeatureExperimentResult:
    """Evaluate handcrafted features with subject-wise grouped cross-validation."""
    original_size = len(windows)
    label_ids = np.unique(labels)
    X, y, clean_groups = extract_feature_dataset(
        windows,
        labels,
        groups,
        threshold=energy_threshold,
        feature_set=feature_set,
    )
    if len(X) == 0:
        raise ValueError("Feature matrix is empty after filtering.")

    unique_groups = np.unique(clean_groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique users for GroupKFold, found {len(unique_groups)}."
        )

    splitter = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        if use_stratified_groups
        else GroupKFold(n_splits=n_splits)
    )
    fold_accuracies: list[float] = []
    first_report = ""
    first_confusion = np.empty((0, 0), dtype=np.int64)

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=clean_groups), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        if model_name == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=random_seed,
            )
        elif model_name == "extra_trees":
            classifier = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=random_seed,
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)
        fold_accuracies.append(accuracy_score(y_val, y_pred) * 100.0)

        if collect_diagnostics and fold_index == 1:
            first_report = classification_report(
                y_val,
                y_pred,
                labels=label_ids,
                digits=4,
                zero_division=0,
            )
            first_confusion = confusion_matrix(y_val, y_pred, labels=label_ids)

    return FeatureExperimentResult(
        original_size=original_size,
        cleaned_size=len(X),
        feature_dim=X.shape[1],
        fold_accuracies=fold_accuracies,
        mean_accuracy=float(np.mean(fold_accuracies)),
        report=first_report,
        confusion=first_confusion,
    )


def run_feature_sweep(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    model_names: list[str],
    feature_sets: list[str],
    energy_thresholds: list[float],
    estimator_counts: list[int],
    max_depths: list[int | None],
    min_samples_leaf_values: list[int],
    class_weights: list[str | None],
    n_splits: int = 5,
    random_seed: int = 42,
    use_stratified_groups: bool = True,
) -> list[SweepRow]:
    """Run a compact experiment grid for cross-user feature models."""
    results: list[SweepRow] = []

    for (
        model_name,
        feature_set,
        energy_threshold,
        n_estimators,
        max_depth,
        min_samples_leaf,
        class_weight,
    ) in product(
        model_names,
        feature_sets,
        energy_thresholds,
        estimator_counts,
        max_depths,
        min_samples_leaf_values,
        class_weights,
    ):
        experiment = run_feature_baseline(
            windows,
            labels,
            groups,
            energy_threshold=energy_threshold,
            n_splits=n_splits,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_seed=random_seed,
            feature_set=feature_set,
            model_name=model_name,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            collect_diagnostics=False,
            use_stratified_groups=use_stratified_groups,
        )
        results.append(
            SweepRow(
                model_name=model_name,
                feature_set=feature_set,
                energy_threshold=energy_threshold,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                mean_accuracy=experiment.mean_accuracy,
                std_accuracy=float(np.std(experiment.fold_accuracies)),
                cleaned_size=experiment.cleaned_size,
                feature_dim=experiment.feature_dim,
            )
        )

    results.sort(key=lambda row: row.mean_accuracy, reverse=True)
    return results


def build_feature_classifier(
    model_name: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    class_weight: str | None,
    random_seed: int,
):
    """Build a tabular classifier for EMG features."""
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_seed,
        )
    if model_name == "soft_vote":
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_seed,
        )
        et = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_seed,
        )
        return VotingClassifier(
            estimators=[("rf", rf), ("et", et)],
            voting="soft",
            n_jobs=1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def _trial_majority_vote(
    metadata: list[WindowMetadata],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate window predictions into per-trial predictions."""
    grouped: dict[tuple[str, str], dict[str, list[int] | int]] = {}
    for meta, true_label, pred_label in zip(metadata, y_true, y_pred):
        key = (meta.user_name, meta.trial_key)
        if key not in grouped:
            grouped[key] = {"true": int(true_label), "preds": []}
        grouped[key]["preds"].append(int(pred_label))

    trial_true = []
    trial_pred = []
    for trial in grouped.values():
        preds = np.asarray(trial["preds"], dtype=np.int64)
        majority = int(np.argmax(np.bincount(preds)))
        trial_true.append(int(trial["true"]))
        trial_pred.append(majority)

    return np.asarray(trial_true, dtype=np.int64), np.asarray(trial_pred, dtype=np.int64)


def run_train_test_evaluation(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    test_metadata: list[WindowMetadata],
    energy_threshold: float = 0.0,
    feature_set: str = "subject_robust",
    model_name: str = "soft_vote",
    n_estimators: int = 300,
    max_depth: int | None = 20,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
    random_seed: int = 42,
) -> HeldoutEvaluationResult:
    """Train on the training split and evaluate once on the testing split."""
    X_train, y_train, _ = extract_feature_dataset(
        train_windows,
        train_labels,
        train_groups,
        threshold=energy_threshold,
        feature_set=feature_set,
    )
    test_groups = np.asarray([meta.group_id for meta in test_metadata], dtype=np.int64)
    X_test, y_test, _ = extract_feature_dataset(
        test_windows,
        test_labels,
        test_groups,
        threshold=energy_threshold,
        feature_set=feature_set,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train or test feature matrix is empty after filtering.")

    filtered_test_metadata = [
        meta
        for meta, label, window in zip(test_metadata, test_labels, test_windows)
        if not (label != 0 and float(np.mean(np.mean(np.abs(window), axis=1))) < energy_threshold)
    ]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = build_feature_classifier(
        model_name=model_name,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_seed=random_seed,
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    label_ids = np.unique(np.concatenate([y_train, y_test]))
    report = classification_report(y_test, y_pred, labels=label_ids, digits=4, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred, labels=label_ids)
    window_accuracy = accuracy_score(y_test, y_pred) * 100.0

    trial_true, trial_pred = _trial_majority_vote(filtered_test_metadata, y_test, y_pred)
    trial_accuracy = accuracy_score(trial_true, trial_pred) * 100.0

    return HeldoutEvaluationResult(
        train_size=len(X_train),
        test_size=len(X_test),
        feature_dim=X_train.shape[1],
        window_accuracy=window_accuracy,
        trial_accuracy=trial_accuracy,
        report=report,
        confusion=confusion,
        users_evaluated=len(np.unique(train_groups)),
    )


def run_personalized_train_test_evaluation(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_metadata: list[WindowMetadata],
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    test_metadata: list[WindowMetadata],
    energy_threshold: float = 0.0,
    feature_set: str = "subject_robust",
    model_name: str = "soft_vote",
    n_estimators: int = 300,
    max_depth: int | None = 20,
    min_samples_leaf: int = 1,
    class_weight: str | None = "balanced",
    random_seed: int = 42,
) -> HeldoutEvaluationResult:
    """Train one model per user on that user's training split and test on the same user's testing split."""
    train_user_ids = np.asarray([meta.group_id for meta in train_metadata], dtype=np.int64)
    test_user_ids = np.asarray([meta.group_id for meta in test_metadata], dtype=np.int64)
    common_users = sorted(set(train_user_ids.tolist()) & set(test_user_ids.tolist()))
    if not common_users:
        raise ValueError("No overlapping users between train and test splits.")

    all_true = []
    all_pred = []
    all_trial_true = []
    all_trial_pred = []
    feature_dim = 0
    total_train_samples = 0

    for user_id in common_users:
        train_mask = train_user_ids == user_id
        test_mask = test_user_ids == user_id

        user_train_windows = train_windows[train_mask]
        user_train_labels = train_labels[train_mask]
        user_test_windows = test_windows[test_mask]
        user_test_labels = test_labels[test_mask]
        user_test_metadata = [meta for meta, keep in zip(test_metadata, test_mask) if keep]

        dummy_train_groups = np.zeros(len(user_train_windows), dtype=np.int64)
        dummy_test_groups = np.zeros(len(user_test_windows), dtype=np.int64)

        X_train, y_train, _ = extract_feature_dataset(
            user_train_windows,
            user_train_labels,
            dummy_train_groups,
            threshold=energy_threshold,
            feature_set=feature_set,
        )
        X_test, y_test, _ = extract_feature_dataset(
            user_test_windows,
            user_test_labels,
            dummy_test_groups,
            threshold=energy_threshold,
            feature_set=feature_set,
        )
        filtered_user_test_metadata = [
            meta
            for meta, label, window in zip(user_test_metadata, user_test_labels, user_test_windows)
            if not (label != 0 and float(np.mean(np.mean(np.abs(window), axis=1))) < energy_threshold)
        ]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        feature_dim = X_train.shape[1]
        total_train_samples += len(X_train)

        classifier = build_feature_classifier(
            model_name=model_name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_seed=random_seed,
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        trial_true, trial_pred = _trial_majority_vote(filtered_user_test_metadata, y_test, y_pred)
        all_trial_true.extend(trial_true.tolist())
        all_trial_pred.extend(trial_pred.tolist())

    all_true_array = np.asarray(all_true, dtype=np.int64)
    all_pred_array = np.asarray(all_pred, dtype=np.int64)
    all_trial_true_array = np.asarray(all_trial_true, dtype=np.int64)
    all_trial_pred_array = np.asarray(all_trial_pred, dtype=np.int64)
    label_ids = np.unique(np.concatenate([all_true_array, all_pred_array]))

    return HeldoutEvaluationResult(
        train_size=total_train_samples,
        test_size=len(all_true_array),
        feature_dim=feature_dim,
        window_accuracy=accuracy_score(all_true_array, all_pred_array) * 100.0,
        trial_accuracy=accuracy_score(all_trial_true_array, all_trial_pred_array) * 100.0,
        report=classification_report(
            all_true_array,
            all_pred_array,
            labels=label_ids,
            digits=4,
            zero_division=0,
        ),
        confusion=confusion_matrix(all_true_array, all_pred_array, labels=label_ids),
        users_evaluated=len(common_users),
    )


def _sequence_trial_accuracy(
    metadata: list[SequenceMetadata],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    grouped: dict[tuple[str, str], tuple[int, int]] = {}
    correct = 0
    total = 0
    for meta, true_label, pred_label in zip(metadata, y_true, y_pred):
        total += 1
        if int(true_label) == int(pred_label):
            correct += 1
        grouped[(meta.user_name, meta.trial_key)] = (int(true_label), int(pred_label))
    if total == 0:
        return 0.0
    return (correct / total) * 100.0


def _fit_sequence_normalization(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel RMS feature normalization statistics from training sequences."""
    stacked = np.concatenate(sequences, axis=0)
    mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
    std = stacked.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _apply_sequence_normalization(
    sequences: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> list[np.ndarray]:
    """Normalize each sequence with train-derived per-channel statistics."""
    return [((sequence - mean) / std).astype(np.float32) for sequence in sequences]


def run_sequence_lstm_heldout_evaluation(
    train_sequences: list[np.ndarray],
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    test_sequences: list[np.ndarray],
    test_labels: np.ndarray,
    test_metadata: list[SequenceMetadata],
    batch_size: int = 1000,
    learning_rate: float = 1e-4,
    step_size: int = 5,
    gamma: float = 0.9,
    early_stopping_patience: int = 5,
    max_epochs: int = 30,
    hidden_size: int = 128,
    num_layers: int = 3,
    device: str | None = None,
    random_seed: int = 42,
    rms_window_len: int = 5,
    rms_stride: int = 5,
) -> SequenceHeldoutResult:
    """Train a paper-style LSTM using train users and evaluate on test users."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
    train_index, val_index = next(splitter.split(train_sequences, train_labels, groups=train_groups))

    train_subset = [train_sequences[index] for index in train_index]
    val_subset = [train_sequences[index] for index in val_index]
    y_train = train_labels[train_index]
    y_val = train_labels[val_index]

    mean, std = _fit_sequence_normalization(train_subset)
    train_subset = _apply_sequence_normalization(train_subset, mean, std)
    val_subset = _apply_sequence_normalization(val_subset, mean, std)
    normalized_test_sequences = _apply_sequence_normalization(test_sequences, mean, std)

    train_loader = DataLoader(
        SequenceTensorDataset(train_subset, y_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_padded_sequences,
    )
    val_loader = DataLoader(
        SequenceTensorDataset(val_subset, y_val),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )
    test_loader = DataLoader(
        SequenceTensorDataset(normalized_test_sequences, test_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_padded_sequences,
    )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EMGSequenceLSTM(
        input_size=8,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(GESTURE_MAP),
    ).to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_state = None
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        for sequences, lengths, labels in train_loader:
            sequences = sequences.to(resolved_device)
            lengths = lengths.to(resolved_device)
            labels = labels.to(resolved_device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for sequences, lengths, labels in val_loader:
                outputs = model(sequences.to(resolved_device), lengths.to(resolved_device))
                val_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.numpy())

        val_accuracy = accuracy_score(np.asarray(val_targets), np.asarray(val_predictions)) * 100.0
        if val_accuracy > best_val_accuracy + 0.1:
            best_val_accuracy = val_accuracy
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(resolved_device)
    model.eval()

    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for sequences, lengths, labels in test_loader:
            outputs = model(sequences.to(resolved_device), lengths.to(resolved_device))
            test_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            test_targets.extend(labels.numpy())

    test_targets_array = np.asarray(test_targets, dtype=np.int64)
    test_predictions_array = np.asarray(test_predictions, dtype=np.int64)
    label_ids = np.unique(np.concatenate([train_labels, test_labels]))
    report = classification_report(
        test_targets_array,
        test_predictions_array,
        labels=label_ids,
        digits=4,
        zero_division=0,
    )
    confusion = confusion_matrix(test_targets_array, test_predictions_array, labels=label_ids)

    return SequenceHeldoutResult(
        train_sequences=len(train_sequences),
        test_sequences=len(test_sequences),
        window_size=rms_window_len,
        stride=rms_stride,
        best_val_accuracy=best_val_accuracy,
        test_accuracy=accuracy_score(test_targets_array, test_predictions_array) * 100.0,
        test_trial_accuracy=_sequence_trial_accuracy(test_metadata, test_targets_array, test_predictions_array),
        report=report,
        confusion=confusion,
    )


def run_heldout_sweep(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    train_metadata: list[WindowMetadata],
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    test_metadata: list[WindowMetadata],
    modes: list[str],
    feature_sets: list[str],
    model_names: list[str],
    energy_thresholds: list[float],
    estimator_counts: list[int],
    max_depths: list[int | None],
    min_samples_leaf_values: list[int],
    class_weights: list[str | None],
    random_seed: int = 42,
) -> list[HeldoutSweepRow]:
    """Run a compact search directly against the held-out train/test split."""
    rows: list[HeldoutSweepRow] = []

    for (
        mode,
        feature_set,
        model_name,
        energy_threshold,
        n_estimators,
        max_depth,
        min_samples_leaf,
        class_weight,
    ) in product(
        modes,
        feature_sets,
        model_names,
        energy_thresholds,
        estimator_counts,
        max_depths,
        min_samples_leaf_values,
        class_weights,
    ):
        if mode == "global":
            result = run_train_test_evaluation(
                train_windows=train_windows,
                train_labels=train_labels,
                train_groups=train_groups,
                test_windows=test_windows,
                test_labels=test_labels,
                test_metadata=test_metadata,
                energy_threshold=energy_threshold,
                feature_set=feature_set,
                model_name=model_name,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_seed=random_seed,
            )
        elif mode == "personalized":
            result = run_personalized_train_test_evaluation(
                train_windows=train_windows,
                train_labels=train_labels,
                train_metadata=train_metadata,
                test_windows=test_windows,
                test_labels=test_labels,
                test_metadata=test_metadata,
                energy_threshold=energy_threshold,
                feature_set=feature_set,
                model_name=model_name,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_seed=random_seed,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rows.append(
            HeldoutSweepRow(
                mode=mode,
                feature_set=feature_set,
                model_name=model_name,
                energy_threshold=energy_threshold,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                feature_dim=result.feature_dim,
                window_accuracy=result.window_accuracy,
                trial_accuracy=result.trial_accuracy,
                users_evaluated=result.users_evaluated,
            )
        )

    rows.sort(key=lambda row: (row.trial_accuracy, row.window_accuracy), reverse=True)
    return rows


def run_cnn_baseline(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.002,
    device: str | None = None,
) -> CNNExperimentResult:
    """Train a baseline 1D CNN with a user-wise train/validation split."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(windows, labels, groups=groups))

    X_train = windows[train_idx]
    X_val = windows[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    X_train, mean, std = standardize_windows(X_train)
    X_val, _, _ = standardize_windows(X_val, mean=mean, std=std)

    train_loader = DataLoader(WindowTensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowTensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EMG1DCNN(num_classes=len(np.unique(labels))).to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    train_loss_history: list[float] = []
    val_accuracy_history: list[float] = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(resolved_device)
            y_batch = y_batch.to(resolved_device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(len(train_loader), 1)
        train_loss_history.append(epoch_loss)
        scheduler.step(epoch_loss)

        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.to(resolved_device))
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(y_batch.numpy())

        predictions_array = np.asarray(predictions)
        targets_array = np.asarray(targets)
        val_accuracy_history.append(accuracy_score(targets_array, predictions_array) * 100.0)

    report = classification_report(targets_array, predictions_array, digits=4)
    return CNNExperimentResult(
        train_loss_history=train_loss_history,
        val_accuracy_history=val_accuracy_history,
        report=report,
    )
