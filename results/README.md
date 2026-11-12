# Historical aggregate results

These tables contain experiment configurations and aggregate metrics, not participant-level records.

- `sequence_full_run.csv`: final row recovered from `sequence_heldout_results.csv`. Its original header had 22 columns while the final row had 24. The 24-column schema was reconstructed from the current CLI logger; `max_trials_per_gesture_per_user` and `apply_filter` account for the added fields. Earlier inconsistent rows were excluded. The final row records 306 training users, 306 testing users, 45,900 trials per split, 81.4516% validation accuracy and 80.2985% test accuracy.
- `heldout_results_v2.csv`: consistent 40-user tree-model development runs.
- `heldout_sweep_results_12u.csv`: exploratory supplied-test-split tuning, not a final unbiased benchmark.
- `sweep_smoke.csv` and `sweep_subject_robust.csv`: historical feature cross-validation sweeps. Their CSV schema omits some run context such as selected user IDs; treat them as exploratory records.

All accuracy values are percentages. No full training run was repeated for this release. The original logs remain unchanged in the original project folder.
