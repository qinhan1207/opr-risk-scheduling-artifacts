# Java Scheduler Services

This folder contains the Java-side integration code used by the scheduling
chain. The code is organized as selected files from the original multi-module
Spring Boot project:

```text
local-scheduler-agent/
global-scheduler/
scheduler-common/
scheduler-pojo/
```

The files are provided to document and reproduce the Java-side behavior of the
Go-Java-Python risk-aware scoring chain. They are not intended to be a complete
standalone Spring Boot release; deployment-specific build files, credentials,
and environment wiring are intentionally excluded.

## Current Model Naming

The current online predictor is named:

```text
OPR-TSMixer-v2
```

Older architecture-specific names have been removed from the Java integration
path. The Java side now treats the Python service as an OPR-TSMixer risk
inference service.

## Runtime Flow

```text
Local Scheduler Agent
  -> collects cross-cluster latency/loss observations
  -> maintains latencyWindow
  -> reports ClusterStatus to Global Scheduler

Global Scheduler
  -> receives ClusterStatus
  -> calls OprRiskPredictor
  -> sends latency_window to Python /predict/window
  -> receives risk probability
  -> computes scheduler-consumable ClusterScore
```

## Key Files

- `local-scheduler-agent/.../ClusterMonitorServiceImpl.java`: collects raw
  cross-cluster observations and aggregates latency/loss.
- `local-scheduler-agent/.../ReportServiceImpl.java`: maintains the recent
  `latencyWindow` and reports it to the global service.
- `global-scheduler/.../NetworkStabilityServiceImpl.java`: calls the Python
  OPR-TSMixer service and fuses current latency with predicted risk.
- `global-scheduler/.../ClusterScoreController.java`: exposes the scoring
  endpoint consumed by the Karmada-side Go extension.
- `scheduler-common/.../OprRiskPredictor.java`: HTTP client for
  `POST /predict/window`.
- `scheduler-pojo/.../ClusterStatus.java`: shared status payload containing
  `latencyWindow`.

## Python Service Contract

The Java client sends:

```json
{
  "cluster_name": "member1",
  "latency_window": [30.1, 31.2, 29.8, 34.0, 36.5, 39.2, 40.1, 41.6, 43.2, 44.0]
}
```

to:

```text
http://127.0.0.1:5001/predict/window
```

The Python service constructs the 13-dimensional OPR feature sequence and
returns:

```json
{
  "prob": 0.123456,
  "is_fault": false,
  "model": "OPR-TSMixer-v2"
}
```

The OPR feature mapping is documented in
`../../docs/opr_feature_mapping.md`.
