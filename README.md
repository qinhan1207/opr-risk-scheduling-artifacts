# OPR Risk Scheduling Artifacts

This repository contains the data and code artifacts for the paper
"Predictive Risk-Aware Multi-Cluster Scheduling for Edge-Cloud Continuum".

The artifacts support the short-term placement-risk prediction and
risk-aware scheduling experiments reported in the paper. The released traces
are collected observations from the controlled multi-cluster testbed.

## Repository Contents

- `data/raw_trace/`: timestamp-level collected latency traces.
- `data/window_samples/`: supervised window-sample metadata used for risk
  prediction analysis.
- `data/workload/`: k6 workload replay result files for ROS, SNS, and RAS
  scheduling strategies.
- `training_scripts/`: model definitions, feature construction, training, and
  evaluation scripts for the prediction experiments.
- `inference_service/`: Python OPR inference service used by the online
  scheduler scoring chain.
- `models/opr-risk/`: trained OPR risk-prediction model artifacts loaded by
  the inference service.
- `scheduler-plugin/`: Go, Java, and integration notes for the risk-aware
  scheduler scoring chain used in the testbed.
- `configs/`: testbed configuration notes, including software versions and
  member-cluster roles.
- `docs/`: implementation notes that clarify artifact-to-paper mappings.
- `scripts/`: controlled network perturbation utility used in the testbed.
- `results/tables/`: CSV summaries corresponding to the main tables and
  figure data reported in the paper.

## Data Summary

The released raw traces contain 104,728 timestamp-level observations:

| Split | File | Rows |
| --- | --- | ---: |
| train | `data/raw_trace/train_v3_raw_trace.csv` | 69,808 |
| test | `data/raw_trace/test_v3_raw_trace.csv` | 34,920 |

The released window-sample metadata contains 85,828 supervised samples:

| Split | File | Rows |
| --- | --- | ---: |
| train | `data/window_samples/train_v3.meta.csv` | 57,208 |
| test | `data/window_samples/test_v3.meta.csv` | 28,620 |

See `data/README.md` for field descriptions and access notes.

## Training and Evaluation Code

The main prediction scripts are in `training_scripts/`. They include the OPR
feature construction logic, OPR-GRU, OPR-ModernTCN, OPR-TSMixer, OPR-xLSTM,
feature-representation comparison, OPR feature-group ablation, and multi-seed
comparison.

The scripts are provided to make the training and evaluation pipeline
inspectable and rerunnable. Re-trained metrics may show minor numerical
variation because of hardware, software, and random initialization differences.

The implementation-level OPR feature order and its mapping to the paper-level
cue groups are documented in `docs/opr_feature_mapping.md`.

## Online Inference Service

The online model service is provided in `inference_service/`. It receives a
recent latency window from the Java global scheduler, constructs OPR features
inside Python, loads `models/opr-risk/model.pt`, and returns a risk
probability for scheduler scoring.

```bash
pip install -r inference_service/requirements.txt
python inference_service/inference_server_opr.py
```

## Paper Result Mapping

| Paper item | Repository artifact |
| --- | --- |
| Table 5 | `results/tables/table5_representation_comparison.csv` |
| Table 6 | `results/tables/table6_opr_group_ablation.csv` |
| Table 7 | `results/tables/table7_predictor_comparison.csv` |
| Table 8 | `results/tables/table8_multiseed_stability.csv` |
| Table 9 | `results/tables/table9_end_to_end_performance.csv` |
| Fig. 4 | `results/tables/figure4_selection_counts.csv` |
| Fig. 5 | `results/tables/figure5_candidate_benchmark.csv` |
| Fig. 6 | `results/tables/table9_end_to_end_performance.csv` and workload JSON files |

## Citation

If you use these artifacts, please cite the corresponding paper. A formal
citation entry will be added after publication.

## License

This repository is released under the Apache License 2.0. The released
datasets and model artifacts are provided for academic research and
reproducibility.
