"""Command-line entry points for the EMG project."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import DEFAULT_TESTING_PATH, DEFAULT_TRAINING_PATH
from .dataset import load_window_dataset, load_window_dataset_with_metadata
from .experiments import (
    run_sequence_lstm_heldout_evaluation,
    run_heldout_sweep,
    run_cnn_baseline,
    run_feature_baseline,
    run_feature_sweep,
    run_personalized_train_test_evaluation,
    run_train_test_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EMG gesture recognition pipeline")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_TRAINING_PATH),
        help="Path to the training JSON directory",
    )
    parser.add_argument(
        "--test-data-dir",
        default=str(DEFAULT_TESTING_PATH),
        help="Path to the testing JSON directory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Load the dataset and print summary stats")
    inspect_parser.add_argument("--window-len", type=int, default=100)
    inspect_parser.add_argument("--overlap", type=int, default=20)
    inspect_parser.add_argument("--max-users", type=int, default=None)
    inspect_parser.add_argument("--sample-users", type=int, default=None)
    inspect_parser.add_argument("--max-windows-per-user", type=int, default=None)
    inspect_parser.add_argument("--seed", type=int, default=42)

    feature_parser = subparsers.add_parser("train-features", help="Run the handcrafted feature baseline")
    feature_parser.add_argument("--window-len", type=int, default=100)
    feature_parser.add_argument("--overlap", type=int, default=20)
    feature_parser.add_argument("--energy-threshold", type=float, default=0.0)
    feature_parser.add_argument("--max-users", type=int, default=None)
    feature_parser.add_argument("--sample-users", type=int, default=None)
    feature_parser.add_argument("--max-windows-per-user", type=int, default=None)
    feature_parser.add_argument("--folds", type=int, default=5)
    feature_parser.add_argument("--trees", type=int, default=100)
    feature_parser.add_argument("--max-depth", type=int, default=15)
    feature_parser.add_argument("--min-samples-leaf", type=int, default=1)
    feature_parser.add_argument(
        "--feature-set",
        choices=["base", "extended", "hybrid", "subject_robust"],
        default="base",
    )
    feature_parser.add_argument(
        "--model",
        choices=["random_forest", "extra_trees"],
        default="random_forest",
    )
    feature_parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "balanced_subsample"],
        default="none",
    )
    feature_parser.add_argument("--seed", type=int, default=42)
    feature_parser.add_argument(
        "--splitter",
        choices=["stratified_group", "group"],
        default="stratified_group",
    )

    sweep_parser = subparsers.add_parser("sweep-features", help="Run a small cross-user feature sweep")
    sweep_parser.add_argument("--window-len", type=int, default=100)
    sweep_parser.add_argument("--overlap", type=int, default=20)
    sweep_parser.add_argument("--max-users", type=int, default=None)
    sweep_parser.add_argument("--sample-users", type=int, default=30)
    sweep_parser.add_argument("--max-windows-per-user", type=int, default=1000)
    sweep_parser.add_argument("--folds", type=int, default=5)
    sweep_parser.add_argument("--seed", type=int, default=42)
    sweep_parser.add_argument(
        "--splitter",
        choices=["stratified_group", "group"],
        default="stratified_group",
    )
    sweep_parser.add_argument(
        "--output",
        default="feature_sweep_results.csv",
        help="CSV file to write sweep results to",
    )

    heldout_parser = subparsers.add_parser(
        "train-heldout",
        help="Train on the training split and evaluate on the testing split",
    )
    heldout_parser.add_argument("--window-len", type=int, default=100)
    heldout_parser.add_argument("--overlap", type=int, default=20)
    heldout_parser.add_argument("--energy-threshold", type=float, default=1.8)
    heldout_parser.add_argument("--max-users", type=int, default=None)
    heldout_parser.add_argument("--sample-users", type=int, default=None)
    heldout_parser.add_argument("--max-windows-per-user", type=int, default=None)
    heldout_parser.add_argument("--trees", type=int, default=300)
    heldout_parser.add_argument("--max-depth", type=int, default=20)
    heldout_parser.add_argument("--min-samples-leaf", type=int, default=1)
    heldout_parser.add_argument(
        "--feature-set",
        choices=["base", "extended", "hybrid", "subject_robust"],
        default="subject_robust",
    )
    heldout_parser.add_argument(
        "--model",
        choices=["random_forest", "extra_trees", "soft_vote"],
        default="soft_vote",
    )
    heldout_parser.add_argument(
        "--class-weight",
        choices=["none", "balanced", "balanced_subsample"],
        default="balanced",
    )
    heldout_parser.add_argument("--seed", type=int, default=42)
    heldout_parser.add_argument(
        "--log-file",
        default="heldout_results.csv",
        help="CSV file to append held-out evaluation results to",
    )
    heldout_parser.add_argument(
        "--mode",
        choices=["global", "personalized"],
        default="global",
        help="Global trains one model across all training users; personalized trains one model per user.",
    )

    heldout_sweep_parser = subparsers.add_parser(
        "sweep-heldout",
        help="Search directly on the held-out training/testing split",
    )
    heldout_sweep_parser.add_argument("--window-len", type=int, default=100)
    heldout_sweep_parser.add_argument("--overlap", type=int, default=20)
    heldout_sweep_parser.add_argument("--max-users", type=int, default=None)
    heldout_sweep_parser.add_argument("--sample-users", type=int, default=20)
    heldout_sweep_parser.add_argument("--max-windows-per-user", type=int, default=1000)
    heldout_sweep_parser.add_argument("--seed", type=int, default=42)
    heldout_sweep_parser.add_argument(
        "--output",
        default="heldout_sweep_results.csv",
        help="CSV file to write held-out sweep results to",
    )

    cnn_parser = subparsers.add_parser("train-cnn", help="Run the 1D CNN baseline")
    cnn_parser.add_argument("--window-len", type=int, default=100)
    cnn_parser.add_argument("--overlap", type=int, default=20)
    cnn_parser.add_argument("--epochs", type=int, default=10)
    cnn_parser.add_argument("--batch-size", type=int, default=128)
    cnn_parser.add_argument("--max-users", type=int, default=None)
    cnn_parser.add_argument("--sample-users", type=int, default=None)
    cnn_parser.add_argument("--max-windows-per-user", type=int, default=None)
    cnn_parser.add_argument("--seed", type=int, default=42)

    sequence_parser = subparsers.add_parser(
        "train-sequence-heldout",
        help="Train the paper-style RMS sequence LSTM on training users and test on testing users",
    )
    sequence_parser.add_argument("--rms-window-len", type=int, default=5)
    sequence_parser.add_argument("--rms-stride", type=int, default=5)
    sequence_parser.add_argument("--max-users", type=int, default=None)
    sequence_parser.add_argument("--sample-users", type=int, default=None)
    sequence_parser.add_argument("--max-trials-per-user", type=int, default=None)
    sequence_parser.add_argument("--max-trials-per-gesture-per-user", type=int, default=None)
    sequence_parser.add_argument("--batch-size", type=int, default=1000)
    sequence_parser.add_argument(
        "--apply-filter",
        action="store_true",
        help="Apply the band-pass filter before RMS sequence extraction. Off by default to match the paper summary.",
    )
    sequence_parser.add_argument("--learning-rate", type=float, default=1e-4)
    sequence_parser.add_argument("--step-size", type=int, default=5)
    sequence_parser.add_argument("--gamma", type=float, default=0.9)
    sequence_parser.add_argument("--early-stopping-patience", type=int, default=5)
    sequence_parser.add_argument("--max-epochs", type=int, default=30)
    sequence_parser.add_argument("--hidden-size", type=int, default=128)
    sequence_parser.add_argument("--num-layers", type=int, default=3)
    sequence_parser.add_argument("--seed", type=int, default=42)
    sequence_parser.add_argument(
        "--log-file",
        default="sequence_heldout_results.csv",
        help="CSV file to append sequence model results to",
    )

    return parser


def maybe_limit_users(windows, labels, groups, max_users):
    if max_users is None:
        return windows, labels, groups
    mask = groups < max_users
    return windows[mask], labels[mask], groups[mask]


def normalize_class_weight(value: str) -> str | None:
    if value == "none":
        return None
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        windows, labels, groups, summary = load_window_dataset(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=getattr(args, "max_users", None),
            sample_users=getattr(args, "sample_users", None),
            max_windows_per_user=getattr(args, "max_windows_per_user", None),
            random_seed=getattr(args, "seed", 42),
        )
        if len(windows) == 0:
            raise SystemExit("No valid EMG windows were loaded.")
        print(f"Users: {summary.users}")
        print(f"Trials: {summary.trials}")
        print(f"Windows: {summary.windows}")
        print(f"Window tensor shape: {windows.shape}")
        print(f"Label distribution: {summary.labels}")
        print(f"Unique groups: {len(np.unique(groups))}")
        print(f"Selected users: {summary.user_names[:10]}{'...' if len(summary.user_names) > 10 else ''}")
        return

    if args.command == "train-features":
        windows, labels, groups, summary = load_window_dataset(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=getattr(args, "max_users", None),
            sample_users=getattr(args, "sample_users", None),
            max_windows_per_user=getattr(args, "max_windows_per_user", None),
            random_seed=getattr(args, "seed", 42),
        )
        if len(windows) == 0:
            raise SystemExit("No valid EMG windows were loaded.")
        result = run_feature_baseline(
            windows,
            labels,
            groups,
            energy_threshold=args.energy_threshold,
            n_splits=args.folds,
            n_estimators=args.trees,
            max_depth=args.max_depth,
            random_seed=args.seed,
            feature_set=args.feature_set,
            model_name=args.model,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=normalize_class_weight(args.class_weight),
            use_stratified_groups=(args.splitter == "stratified_group"),
        )
        print(
            f"🧠 Extracting advanced feature set (MAV, RMS, WL, SSC, ZC) with filter < {args.energy_threshold}..."
        )
        print(f"📊 Original Subset Size: {result.original_size} | Cleaned Dataset Size: {result.cleaned_size}")
        print(f"📐 Feature Matrix Shape: ({result.cleaned_size}, {result.feature_dim})")
        print(f"👥 Active Users In Run: {summary.users}")
        print(
            f"🌲 Model Config: model={args.model}, feature_set={args.feature_set}, "
            f"trees={args.trees}, max_depth={args.max_depth}, min_samples_leaf={args.min_samples_leaf}, "
            f"class_weight={args.class_weight}, folds={args.folds}, splitter={args.splitter}"
        )
        print(f"🪟 Max Windows Per User: {args.max_windows_per_user}")
        print("Feature baseline fold accuracies:", [round(value, 2) for value in result.fold_accuracies])
        print(f"\n🎯 Cross-User Mean Accuracy: {result.mean_accuracy:.2f}%")
        print("\n📊 Detailed Classification Report:")
        print(result.report)
        print("🧩 First Fold Confusion Matrix:")
        print(result.confusion)
        return

    if args.command == "sweep-features":
        windows, labels, groups, _ = load_window_dataset(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=getattr(args, "max_users", None),
            sample_users=getattr(args, "sample_users", None),
            max_windows_per_user=getattr(args, "max_windows_per_user", None),
            random_seed=getattr(args, "seed", 42),
        )
        if len(windows) == 0:
            raise SystemExit("No valid EMG windows were loaded.")
        sweep_rows = run_feature_sweep(
            windows,
            labels,
            groups,
            model_names=["random_forest", "extra_trees"],
            feature_sets=["base", "extended", "hybrid", "subject_robust"],
            energy_thresholds=[1.0, 1.8],
            estimator_counts=[200],
            max_depths=[15, 20],
            min_samples_leaf_values=[1, 3],
            class_weights=[None, "balanced"],
            n_splits=args.folds,
            random_seed=args.seed,
            use_stratified_groups=(args.splitter == "stratified_group"),
        )

        output_path = Path(args.output)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "model_name",
                    "feature_set",
                    "energy_threshold",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "class_weight",
                    "mean_accuracy",
                    "std_accuracy",
                    "cleaned_size",
                    "feature_dim",
                ],
            )
            writer.writeheader()
            for rank, row in enumerate(sweep_rows, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "model_name": row.model_name,
                        "feature_set": row.feature_set,
                        "energy_threshold": row.energy_threshold,
                        "n_estimators": row.n_estimators,
                        "max_depth": row.max_depth,
                        "min_samples_leaf": row.min_samples_leaf,
                        "class_weight": row.class_weight,
                        "mean_accuracy": round(row.mean_accuracy, 4),
                        "std_accuracy": round(row.std_accuracy, 4),
                        "cleaned_size": row.cleaned_size,
                        "feature_dim": row.feature_dim,
                    }
                )

        print(f"Completed {len(sweep_rows)} sweep runs.")
        print(f"Saved results to {output_path}")
        print("Top 5 configurations:")
        for rank, row in enumerate(sweep_rows[:5], start=1):
            print(
                f"{rank}. acc={row.mean_accuracy:.2f}% | model={row.model_name} | "
                f"features={row.feature_set} | threshold={row.energy_threshold} | "
                f"trees={row.n_estimators} | max_depth={row.max_depth} | "
                f"min_leaf={row.min_samples_leaf} | class_weight={row.class_weight}"
            )
        return

    if args.command == "train-heldout":
        train_X, train_y, train_groups, train_metadata, train_summary = load_window_dataset_with_metadata(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_windows_per_user=args.max_windows_per_user,
            random_seed=args.seed,
        )
        test_X, test_y, _, test_metadata, test_summary = load_window_dataset_with_metadata(
            Path(args.test_data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_windows_per_user=args.max_windows_per_user,
            random_seed=args.seed,
        )
        if args.mode == "global":
            result = run_train_test_evaluation(
                train_X,
                train_y,
                train_groups,
                test_X,
                test_y,
                test_metadata,
                energy_threshold=args.energy_threshold,
                feature_set=args.feature_set,
                model_name=args.model,
                n_estimators=args.trees,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight=normalize_class_weight(args.class_weight),
                random_seed=args.seed,
            )
        else:
            result = run_personalized_train_test_evaluation(
                train_X,
                train_y,
                train_metadata,
                test_X,
                test_y,
                test_metadata,
                energy_threshold=args.energy_threshold,
                feature_set=args.feature_set,
                model_name=args.model,
                n_estimators=args.trees,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                class_weight=normalize_class_weight(args.class_weight),
                random_seed=args.seed,
            )

        log_path = Path(args.log_file)
        row = {
            "feature_set": args.feature_set,
            "model": args.model,
            "energy_threshold": args.energy_threshold,
            "trees": args.trees,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": args.class_weight,
            "window_len": args.window_len,
            "overlap": args.overlap,
            "max_users": args.max_users,
            "sample_users": args.sample_users,
            "max_windows_per_user": args.max_windows_per_user,
            "train_users": train_summary.users,
            "test_users": test_summary.users,
            "train_size": result.train_size,
            "test_size": result.test_size,
            "feature_dim": result.feature_dim,
            "window_accuracy": round(result.window_accuracy, 4),
            "trial_accuracy": round(result.trial_accuracy, 4),
            "mode": args.mode,
            "users_evaluated": result.users_evaluated,
            "seed": args.seed,
        }
        file_exists = log_path.exists()
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print("Held-out evaluation complete.")
        print(
            f"Mode: {args.mode} | Train users: {train_summary.users} | "
            f"Test users: {test_summary.users} | Evaluated users: {result.users_evaluated} | "
            f"Feature dim: {result.feature_dim}"
        )
        print(
            f"Window accuracy: {result.window_accuracy:.2f}% | "
            f"Trial accuracy: {result.trial_accuracy:.2f}%"
        )
        print("\nClassification report:")
        print(result.report)
        print("Confusion matrix:")
        print(result.confusion)
        print(f"Logged result to {log_path}")
        return

    if args.command == "sweep-heldout":
        train_X, train_y, train_groups, train_metadata, train_summary = load_window_dataset_with_metadata(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_windows_per_user=args.max_windows_per_user,
            random_seed=args.seed,
        )
        test_X, test_y, _, test_metadata, test_summary = load_window_dataset_with_metadata(
            Path(args.test_data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_windows_per_user=args.max_windows_per_user,
            random_seed=args.seed,
        )
        rows = run_heldout_sweep(
            train_windows=train_X,
            train_labels=train_y,
            train_groups=train_groups,
            train_metadata=train_metadata,
            test_windows=test_X,
            test_labels=test_y,
            test_metadata=test_metadata,
            modes=["global"],
            feature_sets=["hybrid", "subject_robust"],
            model_names=["extra_trees", "random_forest"],
            energy_thresholds=[1.0, 1.8],
            estimator_counts=[300, 500],
            max_depths=[20, None],
            min_samples_leaf_values=[1, 3],
            class_weights=["balanced"],
            random_seed=args.seed,
        )

        output_path = Path(args.output)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "mode",
                    "feature_set",
                    "model_name",
                    "energy_threshold",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "class_weight",
                    "feature_dim",
                    "window_accuracy",
                    "trial_accuracy",
                    "users_evaluated",
                    "train_users",
                    "test_users",
                ],
            )
            writer.writeheader()
            for rank, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "mode": row.mode,
                        "feature_set": row.feature_set,
                        "model_name": row.model_name,
                        "energy_threshold": row.energy_threshold,
                        "n_estimators": row.n_estimators,
                        "max_depth": row.max_depth,
                        "min_samples_leaf": row.min_samples_leaf,
                        "class_weight": row.class_weight,
                        "feature_dim": row.feature_dim,
                        "window_accuracy": round(row.window_accuracy, 4),
                        "trial_accuracy": round(row.trial_accuracy, 4),
                        "users_evaluated": row.users_evaluated,
                        "train_users": train_summary.users,
                        "test_users": test_summary.users,
                    }
                )

        print(f"Completed {len(rows)} held-out runs.")
        print(f"Saved results to {output_path}")
        print("Top 5 configurations by trial accuracy:")
        for rank, row in enumerate(rows[:5], start=1):
            print(
                f"{rank}. trial={row.trial_accuracy:.2f}% | window={row.window_accuracy:.2f}% | "
                f"features={row.feature_set} | model={row.model_name} | threshold={row.energy_threshold} | "
                f"trees={row.n_estimators} | max_depth={row.max_depth} | "
                f"min_leaf={row.min_samples_leaf} | class_weight={row.class_weight}"
            )
        return

    if args.command == "train-cnn":
        windows, labels, groups, _ = load_window_dataset(
            Path(args.data_dir),
            window_len=args.window_len,
            overlap=args.overlap,
            max_users=getattr(args, "max_users", None),
            sample_users=getattr(args, "sample_users", None),
            max_windows_per_user=getattr(args, "max_windows_per_user", None),
            random_seed=getattr(args, "seed", 42),
        )
        if len(windows) == 0:
            raise SystemExit("No valid EMG windows were loaded.")
        result = run_cnn_baseline(
            windows,
            labels,
            groups,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print("Train loss history:", [round(value, 4) for value in result.train_loss_history])
        print("Validation accuracy history:", [round(value, 2) for value in result.val_accuracy_history])
        print(result.report)
        return

    if args.command == "train-sequence-heldout":
        from .dataset import load_rms_sequence_dataset

        train_sequences, train_labels, train_groups, _, train_summary = load_rms_sequence_dataset(
            Path(args.data_dir),
            rms_window_len=args.rms_window_len,
            rms_stride=args.rms_stride,
            apply_filter=args.apply_filter,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_trials_per_user=args.max_trials_per_user,
            max_trials_per_gesture_per_user=args.max_trials_per_gesture_per_user,
            random_seed=args.seed,
        )
        test_sequences, test_labels, _, test_metadata, test_summary = load_rms_sequence_dataset(
            Path(args.test_data_dir),
            rms_window_len=args.rms_window_len,
            rms_stride=args.rms_stride,
            apply_filter=args.apply_filter,
            max_users=args.max_users,
            sample_users=args.sample_users,
            max_trials_per_user=args.max_trials_per_user,
            max_trials_per_gesture_per_user=args.max_trials_per_gesture_per_user,
            random_seed=args.seed,
        )
        result = run_sequence_lstm_heldout_evaluation(
            train_sequences=train_sequences,
            train_labels=train_labels,
            train_groups=train_groups,
            test_sequences=test_sequences,
            test_labels=test_labels,
            test_metadata=test_metadata,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            step_size=args.step_size,
            gamma=args.gamma,
            early_stopping_patience=args.early_stopping_patience,
            max_epochs=args.max_epochs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            random_seed=args.seed,
            rms_window_len=args.rms_window_len,
            rms_stride=args.rms_stride,
        )

        row = {
            "model": "paper_lstm",
            "rms_window_len": args.rms_window_len,
            "rms_stride": args.rms_stride,
            "sample_users": args.sample_users,
            "max_users": args.max_users,
            "max_trials_per_user": args.max_trials_per_user,
            "max_trials_per_gesture_per_user": args.max_trials_per_gesture_per_user,
            "batch_size": args.batch_size,
            "apply_filter": args.apply_filter,
            "learning_rate": args.learning_rate,
            "step_size": args.step_size,
            "gamma": args.gamma,
            "early_stopping_patience": args.early_stopping_patience,
            "max_epochs": args.max_epochs,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "train_users": train_summary.users,
            "test_users": test_summary.users,
            "train_sequences": result.train_sequences,
            "test_sequences": result.test_sequences,
            "best_val_accuracy": round(result.best_val_accuracy, 4),
            "test_accuracy": round(result.test_accuracy, 4),
            "test_trial_accuracy": round(result.test_trial_accuracy, 4),
            "seed": args.seed,
        }
        log_path = Path(args.log_file)
        file_exists = log_path.exists()
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print("Sequence held-out evaluation complete.")
        print(
            f"Train users: {train_summary.users} | Test users: {test_summary.users} | "
            f"Train sequences: {result.train_sequences} | Test sequences: {result.test_sequences}"
        )
        print(
            f"Best validation accuracy: {result.best_val_accuracy:.2f}% | "
            f"Test accuracy: {result.test_accuracy:.2f}% | "
            f"Test trial accuracy: {result.test_trial_accuracy:.2f}%"
        )
        print("\nClassification report:")
        print(result.report)
        print("Confusion matrix:")
        print(result.confusion)
        print(f"Logged result to {log_path}")
        return


if __name__ == "__main__":
    main()
