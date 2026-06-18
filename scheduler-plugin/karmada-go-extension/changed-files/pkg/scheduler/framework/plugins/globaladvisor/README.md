# GlobalAdvisor Scheduler Plugin

`GlobalAdvisor` is a Karmada scheduler framework plugin that calls the Java
global scheduler service during candidate filtering and scoring.

The plugin is designed for the OPR risk-aware scheduling chain:

```text
Karmada scheduler plugin
  -> Java global scheduler /api/advisor/score
  -> Python OPR-TSMixer-v2 inference service
  -> risk-aware candidate score
```

## Extension Points

The plugin can expose both Filter and Score extension points, or only one of
them, controlled by `GLOBAL_ADVISOR_MODE`:

| Value | Behavior |
| --- | --- |
| `both` | expose Filter and Score; default |
| `filter-only` | expose Filter only |
| `score-only` | expose Score only |

## Configuration

Environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `GLOBAL_SCHEDULER_URL` | `http://127.0.0.1:9090` | Java global scheduler base URL |
| `GLOBAL_ADVISOR_MODE` | `both` | enabled extension points |
| `TEST_AFFINITY_TARGET` | empty | optional experiment-only affinity target |

The plugin calls:

```text
GET /api/advisor/score?cluster=<clusterName>
GET /api/advisor/score?cluster=<clusterName>&target=<targetCluster>
```

Expected response:

```json
{
  "clusterName": "member1",
  "healthScore": 87.5,
  "reason": "Latency:30.10ms Risk:0.1200 Cost:0.110 -> Score:89.0"
}
```

## Enabling the Plugin

Example scheduler command:

```bash
karmada-scheduler \
  --kubeconfig=/etc/karmada/config/karmada.config \
  --plugins=APIEnablement,ClusterAffinity,ClusterEviction,SpreadConstraint,TaintToleration,GlobalAdvisor
```

For a local debug run:

```powershell
go run cmd/scheduler/main.go `
  --kubeconfig="E:\karmada-config" `
  --plugins=APIEnablement,TaintToleration,GlobalAdvisor `
  --enable-scheduler-estimator=false `
  --leader-elect=false `
  --metrics-bind-address=0.0.0.0:8080 `
  --health-probe-bind-address=0.0.0.0:10351 `
  --logging-format=text `
  --v=4
```

## Behavior

Filter stage:

- Allows scheduling when the Java service is unavailable.
- Allows scheduling when candidate status data is missing.
- Filters a candidate when the returned score is `<= 55`.

Score stage:

- Calls the Java global scheduler for each candidate cluster.
- Clamps returned scores to `[0, 100]`.
- Uses a short cache for requests without target-cluster context.
- Falls back to score `50` when the Java service is unavailable.

## Registration

The plugin is registered in:

```text
pkg/scheduler/framework/plugins/registry.go
```

with:

```go
globaladvisor.Name: globaladvisor.New,
```
