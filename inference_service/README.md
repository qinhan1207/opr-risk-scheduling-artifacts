# OPR Inference Service

This service exposes the Python-side OPR risk predictor used by the scheduler.
Java sends a recent latency window, and this service builds the OPR feature
sequence before running the trained model.

The 13-dimensional OPR feature order used by this service is documented in
`../docs/opr_feature_mapping.md`. The feature order is fixed by the saved
`model.pt` and `scaler.json` artifacts.

## Model Artifacts

The default model directory is:

```text
models/opr-risk/
```

It must contain:

```text
model.pt
scaler.json
train_meta.json
```

## Run

```bash
pip install -r inference_service/requirements.txt
python inference_service/inference_server_opr.py
```

Environment overrides:

```bash
OPR_MODEL_DIR=models/opr-risk
OPR_INFERENCE_PORT=5001
OPR_TAU_NET_MS=45.0
OPR_ROLLING_WINDOW=10
```

## API

`POST /predict/window`

```json
{
  "cluster_name": "member1",
  "latency_window": [30.1, 31.2, 29.8, 34.0, 36.5, 39.2, 40.1, 41.6, 43.2, 44.0]
}
```

Response:

```json
{
  "code": 200,
  "prob": 0.123456,
  "is_fault": false,
  "threshold": 0.608351052035709,
  "cluster": "member1",
  "mode": "latency_window",
  "model": "OPR-TSMixer-v2"
}
```
