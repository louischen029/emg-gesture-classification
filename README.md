# EMG Gesture Classification

An eight-channel Myo EMG pipeline for six gestures: no gesture, fist, open, pinch, wave in, and wave out. It compares handcrafted feature classifiers with a convolutional baseline and a stacked LSTM over RMS sequences.

## Recorded result

The full LSTM experiment recorded **80.2985% test-trial accuracy** over **45,900 test trials**, with 306 users in each dataset split. This is a historical logged result, not a newly reproduced benchmark. Both local splits contain the same 306 user directory names; this result does **not establish generalization to unseen participants**. The LSTM selects a checkpoint using one subject-grouped validation fold within the training split, then evaluates the supplied testing split.

Use subject-grouped feature cross-validation to measure performance on unseen user groups. Do not compare its scores directly with supplied-split scores. The exploratory `sweep-heldout` command tunes against the testing split, so its scores are development results rather than an independent final benchmark.

## Setup

The existing local environment uses Python 3.9.6 and the package versions pinned below. Use Python 3.9 for the closest match; other Python versions/platforms may need different compatible package builds.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m emg_pipeline.cli --help
python -m unittest discover -s tests -v
```

## Data stays local

No participant records, photos, datasets, or trained weights are distributed here. The original folder did not provide a verified dataset source or redistribution license. Obtain authorized access to the original corpus separately; do not upload participant files to this repository.

Expected layout:

```text
data/
  training/trainingJSON/user1/<record>.json
  testing/testingJSON/user1/<record>.json
```

Each record contains `trainingSamples`, a mapping of trial keys to objects with `gestureName` and `emg`. EMG contains `ch1` through `ch8`, each a numeric array. Channels are clipped to the shortest length. Root-level data path options must precede the command:

```bash
python -m emg_pipeline.cli --data-dir /local/trainingJSON inspect --max-users 2
```

## Run experiments

Inspect a small subset:

```bash
python -m emg_pipeline.cli inspect --max-users 2 --max-windows-per-user 100
```

Subject-grouped feature cross-validation:

```bash
python -m emg_pipeline.cli train-features --sample-users 40 --max-windows-per-user 1200 --energy-threshold 1.8 --folds 5 --feature-set subject_robust --model extra_trees --class-weight balanced --splitter stratified_group --seed 42
```

Full RMS sequence experiment matching the recorded configuration (potentially expensive):

```bash
mkdir -p outputs
python -m emg_pipeline.cli train-sequence-heldout --rms-window-len 5 --rms-stride 5 --batch-size 1000 --learning-rate 0.0001 --step-size 5 --gamma 0.9 --early-stopping-patience 5 --max-epochs 30 --hidden-size 128 --num-layers 3 --seed 42 --log-file outputs/sequence_results.csv
```

RMS sequences use five-sample windows with stride five and no band-pass filter by default. Window-based experiments use a 20–95 Hz Butterworth band-pass at 200 Hz, followed by 100-sample windows with 20-sample overlap. Feature families include time-domain descriptors, spectral descriptors, normalized features, and channel correlations. The energy threshold changes which windows are evaluated and must be reported with results.

## Repository layout

- `emg_pipeline/dataset.py`: local JSON parsing, window and sequence datasets.
- `emg_pipeline/preprocessing.py`: filtering and normalization.
- `emg_pipeline/features.py`: handcrafted feature extraction.
- `emg_pipeline/models.py`: CNN and variable-length sequence LSTM.
- `emg_pipeline/experiments.py`: grouped validation, training, evaluation, sweeps.
- `emg_pipeline/cli.py`: command-line interface.
- `results/`: curated aggregate historical logs and provenance.
- `tests/`: synthetic-data checks; no participant samples.

Root scripts retain the earlier inspect/feature/CNN entry points. The CNN architecture expects 100-sample windows. The experimental personalized path aligns users by independently assigned numeric group IDs and should not be used without verifying participant identity alignment.

## Reproducibility limits

The package versions come from the surviving local environment, not an archived environment from the historical run. No historical checkpoint, dataset checksum, or original hardware record is available. Seeds are exposed, but exact neural training results may vary by hardware and backend. The full 306-user training run has not been rerun during repository preparation. Dataset acquisition instructions and licensing remain to be verified before claiming independent end-to-end reproducibility.
