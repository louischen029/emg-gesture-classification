"""End-to-end loading and model checks using generated signals only."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from emg_pipeline.dataset import load_window_dataset, load_rms_sequence_dataset
from emg_pipeline.experiments import run_feature_baseline
from emg_pipeline.models import EMGSequenceLSTM, collate_padded_sequences, SequenceTensorDataset

class PipelineTests(unittest.TestCase):
    def test_synthetic_json_to_grouped_classifier_and_lstm(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            for user in range(6):
                root = Path(directory) / f"user{user}"
                root.mkdir()
                trials = {}
                for idx, gesture in enumerate(["fist", "open"]):
                    signal = rng.normal(size=(8, 240)) * (idx + 1)
                    trials[f"idx_{idx}"] = {"gestureName": gesture, "emg": {f"ch{c+1}": signal[c].tolist() for c in range(8)}}
                (root / "synthetic.json").write_text(json.dumps({"trainingSamples": trials}))
            X, y, groups, summary = load_window_dataset(directory)
            self.assertEqual(X.shape, (24, 8, 100))
            self.assertEqual(summary.users, 6)
            result = run_feature_baseline(X, y, groups, n_splits=3, n_estimators=5)
            self.assertEqual(len(result.fold_accuracies), 3)
            self.assertTrue(np.isfinite(result.mean_accuracy))
            seq, labels, _, _, summary = load_rms_sequence_dataset(directory)
            self.assertEqual(summary.trials, 12)
            self.assertEqual(seq[0].shape, (48, 8))
            batch = collate_padded_sequences([SequenceTensorDataset(seq, labels)[i] for i in range(2)])
            model = EMGSequenceLSTM(hidden_size=8, num_layers=1)
            logits = model(batch[0], batch[1])
            self.assertEqual(tuple(logits.shape), (2, 6))
            torch.nn.functional.cross_entropy(logits, batch[2]).backward()

if __name__ == "__main__":
    unittest.main()
