# OPR Feature Mapping

This document records the implementation-level OPR feature order used by the
released training scripts, saved model artifacts, and online inference service.
It is intended to make the mapping between the paper-level OPR cue groups and
the executable code explicit.

## Notation

For one member cluster, let `z_t` denote the member-cluster-level latency risk
observation at sampling time `t`. In the online service, this value is the
aggregated latency value received from the Java scheduler service.

Let `tau` be the observation-layer network-risk threshold. The default value is
45 ms. Let `w` be the local rolling window length. The default value is 10.

For each time step, the code builds a 13-dimensional feature vector. A model
input sample is a sliding sequence with shape:

```text
lookback x feature_dim = 10 x 13
```

The saved `model.pt` and `scaler.json` are bound to the feature order below.
Changing this order requires retraining the model and regenerating the scaler.

## Implementation Feature Order

| Index | Code feature name | Definition | OPR cue group |
| ---: | --- | --- | --- |
| 0 | `raw_latency` | `z_t` | Raw/local state |
| 1 | `delta_latency` | `z_t - z_{t-1}`; first value is 0 | Raw/local state |
| 2 | `rolling_mean` | Mean of recent observations in the local window | Raw/local state |
| 3 | `rolling_std` | Standard deviation of recent observations in the local window | Raw/local state |
| 4 | `residual` | `z_t - rolling_mean` | Local deviation |
| 5 | `abs_residual` | `abs(z_t - rolling_mean)` | Local deviation |
| 6 | `distance_to_threshold` | `tau - z_t` | Threshold proximity |
| 7 | `near_threshold_ratio` | `z_t / tau` | Threshold proximity |
| 8 | `pos_delta_latency` | `max(z_t - z_{t-1}, 0)` | Positive growth |
| 9 | `pos_delta_std` | `max(rolling_std_t - rolling_std_{t-1}, 0)` | Positive growth |
| 10 | `rolling_max_ratio` | `max(W_t) / tau` | Local peak |
| 11 | `pos_delta_sum` | Sum of positive latency increments in the local window | Positive growth |
| 12 | `near_threshold_count_ratio` | Fraction of local-window observations satisfying `z >= 0.8 * tau` | Near-threshold persistence |

`W_t` denotes the available local rolling window ending at time `t`. When fewer
than `w` observations are available, the available prefix is used.

## Relation to Paper-Level OPR Cues

The paper describes OPR as an onset-preserving representation that organizes
recent network-risk observations into several cue groups:

```text
local state
local deviation
threshold proximity
positive growth
near-threshold persistence
local peak
```

The implementation instantiates these cue groups with the concrete 13 features
listed above. The groups are compact and partially overlapping:

- Raw/local state: current level, first-order change, local mean, and local
  variation.
- Local deviation: signed and absolute deviation from the local context.
- Threshold proximity: absolute margin to the risk threshold and normalized
  closeness to the threshold.
- Positive growth: one-step positive growth, local accumulated positive growth,
  and positive change in local variability.
- Near-threshold persistence: fraction of recent observations near the risk
  boundary.
- Local peak: recent maximum latency normalized by the threshold.

This is the feature representation used by:

- `training_scripts/build_rote_dataset.py`
- `training_scripts/train_collected_v3.py`
- `models/opr-risk/model.pt`
- `models/opr-risk/scaler.json`
- `inference_service/inference_server_opr.py`

## Online Inference Contract

The online service receives:

```json
{
  "cluster_name": "member1",
  "latency_window": [30.1, 31.2, 29.8, 34.0, 36.5, 39.2, 40.1, 41.6, 43.2, 44.0]
}
```

It then performs the following steps:

```text
latency_window
  -> implementation-level OPR feature construction
  -> z-score normalization using models/opr-risk/scaler.json
  -> OPR-TSMixer-v2 model inference
  -> sigmoid risk probability
```

The returned `prob` is the candidate member-cluster-level short-term risk
probability consumed by the scheduler-side scoring chain.
