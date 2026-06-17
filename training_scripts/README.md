# Training Scripts

This folder collects the data-construction and training scripts used for the
short-term placement-risk prediction experiments in the PRAMS/OPR study. The
released traces are collected experimental observations; this folder only keeps
the scripts needed to construct window samples and train/evaluate the models.

The exact 13-dimensional OPR feature order used by these scripts is documented
in `../docs/opr_feature_mapping.md`. This order is part of the model contract:
the saved scaler and model weights must be used with the same feature order.

## Script map

- `build_rote_dataset.py`: construct OPR window features, risk labels, and
  sample metadata from collected latency traces.
- `train_feature_ablation_v3.py`: train the TSMixer models for the raw latency,
  basic-statistic, and full OPR representation comparison.
- `train_opr_group_ablation_v3.py`: train the OPR feature-group ablation models.
- `train_collected_v3.py`: train the OPR-GRU, OPR-ModernTCN, OPR-TSMixer, and
  OPR-xLSTM predictors used for the model-structure comparison.
- `train_structure_seed_v3.py`: run the multi-seed comparison for OPR-GRU and
  OPR-TSMixer.
- `train_rote_models_round2.py`: shared model definitions, metrics, and training
  utilities used by the v3 scripts.
- `rote_mamba.py`: dependency imported by `train_rote_models_round2.py`.

## Main reproduction commands

```bash
python build_rote_dataset.py
python train_feature_ablation_v3.py --out-dir collected_v3/collected_v3_feature_ablation_normfix_full
python train_opr_group_ablation_v3.py --out-dir collected_v3/collected_v3_opr_group_ablation_full
python train_collected_v3.py --out-dir collected_v3/collected_v3_new_opr_models_full
python train_structure_seed_v3.py --out-dir collected_v3/collected_v3_structure_seeds_full
```
