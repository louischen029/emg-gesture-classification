from emg_pipeline.dataset import load_window_dataset
from emg_pipeline.preprocessing import butter_bandpass_filter, window_signal


def process_signal_windows(raw_emg, window_len=100, overlap=20):
    """Backward-compatible alias for the new windowing helper."""
    return window_signal(raw_emg, window_len=window_len, overlap=overlap)


def extract_user_data(parent_path):
    """Backward-compatible wrapper returning `(X, y, groups)`."""
    X, y, groups, _ = load_window_dataset(parent_path)
    return X, y, groups


if __name__ == "__main__":
    X, y, groups = extract_user_data("data/training/trainingJSON")
    print(f"✅ Pipeline Verified: Generated raw 3D array matrix shape: {X.shape}")
    print(f"✅ Label matrix shape: {y.shape}")
    print(f"✅ Total Discovered User Groups tracker size: {groups.shape}")
