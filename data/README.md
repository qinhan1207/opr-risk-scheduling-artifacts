# Data

This directory contains collected traces, supervised window-sample metadata,
and workload replay results used in the PRAMS/OPR experiments.

## Raw Traces

Files:

- `raw_trace/train_v3_raw_trace.csv`
- `raw_trace/test_v3_raw_trace.csv`

Columns:

- `t`: timestamp index in the collected trace split.
- `latency_ms`: member-cluster-level latency observation in milliseconds.
- `Raw_Latency`: same latency observation retained for compatibility with the
  training scripts.
- `scenario_id`: contiguous scenario/block identifier.
- `block_type`: controlled perturbation or stable block type.
- `phase`: phase inside the block.
- `onset_type`: risk-transition category for the block.
- `risk_now`: whether the current observation exceeds the network-risk
  threshold.
- `split_name`: trace split name.

## Window Samples

Files:

- `window_samples/train_v3.meta.csv`
- `window_samples/test_v3.meta.csv`

These files describe supervised windows constructed from the collected traces.
The paper uses a historical window length of `L=10`, a future risk horizon of
5 sampling points, and a network-risk threshold of 45 ms.

Key columns:

- `source`: collected split identifier.
- `start_idx`, `end_idx`: window boundaries in the source trace.
- `sample_type`, `state_name`: risk-evolution state of the sample.
- `y_risk`: future placement-risk label.
- `time_to_risk`: distance to the first future high-risk observation.
- `risk_intensity`: normalized future risk intensity.
- `current_normal`: whether the scheduling-time observation is below the risk
  threshold.
- `state_cls`: encoded state class.
- `future_onset_type`, `onset_type`: future and block-level onset categories.
- `future_first_risk_idx`, `future_first_risk_latency`: first high-risk future
  observation, when present.
- `current_latency`, `future_max_latency`: current and future latency summaries.
- `scenario_id`, `block_type`, `phase`: provenance fields inherited from the
  raw trace.

## Workload Replay Results

Files under `workload/` are k6 JSON result files grouped by scheduling
strategy:

- `workload/ros_test/`: resource-oriented scheduling.
- `workload/sns_test/`: snapshot network-aware scheduling.
- `workload/ras_test/`: risk-aware scheduling.

These files support the service-level latency and throughput summaries in
Table 9 and Fig. 6.

