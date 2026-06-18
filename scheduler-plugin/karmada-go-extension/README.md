# Karmada Go Scheduler Extension

This folder contains the Go-side Karmada scheduler extension used in the
OPR risk-aware scheduling chain.

The extension is implemented as an in-tree scheduler framework plugin named
`GlobalAdvisor`. It calls the Java global scheduler service during Karmada
candidate filtering and scoring.

## Source Context

The local development repository was based on a forked Karmada tree:

```text
Repository: D:\code\go\karmada
Branch: master
Commit: d0378dc0c
```

The plugin files are provided under:

```text
changed-files/pkg/scheduler/framework/plugins/globaladvisor/
changed-files/pkg/scheduler/framework/plugins/registry.go
```

Only the scheduler-extension files are included here. The full Karmada source
tree is not duplicated in this artifact repository.

## Plugin Role

```text
Karmada scheduler
  -> GlobalAdvisor plugin
  -> Java global scheduler /api/advisor/score
  -> Python OPR-TSMixer-v2 inference service
  -> candidate risk-aware score
```

The Java service returns a score in `[0, 100]`. The Go plugin consumes this
score in two ways:

- Filter: suppress candidates whose score is low.
- Score: return the Java score as the Karmada candidate score.

If the Java scoring service is unavailable, the plugin falls back to permissive
behavior so the native scheduler remains usable.

## Files

| Path | Purpose |
| --- | --- |
| `globaladvisor/global_advisor.go` | Filter and Score plugin logic |
| `globaladvisor/client.go` | HTTP client for Java `/api/advisor/score` |
| `globaladvisor/cache.go` | short-lived score cache |
| `globaladvisor/types.go` | response DTO |
| `globaladvisor/register.go` | plugin constructor wrapper |
| `registry.go` | in-tree plugin registry entry |

## Apply Conceptually

To reproduce the integration in a Karmada source tree:

1. Copy `changed-files/pkg/scheduler/framework/plugins/globaladvisor/` into
   `pkg/scheduler/framework/plugins/globaladvisor/`.
2. Add the `globaladvisor` import to
   `pkg/scheduler/framework/plugins/registry.go`.
3. Add `globaladvisor.Name: globaladvisor.New` to `NewInTreeRegistry()`.
4. Build and run `karmada-scheduler` with `GlobalAdvisor` enabled.

Example:

```bash
karmada-scheduler \
  --kubeconfig=/etc/karmada/config/karmada.config \
  --plugins=APIEnablement,ClusterAffinity,ClusterEviction,SpreadConstraint,TaintToleration,GlobalAdvisor
```

## Runtime Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `GLOBAL_SCHEDULER_URL` | `http://127.0.0.1:9090` | Java global scheduler base URL |
| `GLOBAL_ADVISOR_MODE` | `both` | `both`, `filter-only`, or `score-only` |
| `TEST_AFFINITY_TARGET` | empty | optional experiment-only target-cluster hint |

## Interface

Request:

```text
GET /api/advisor/score?cluster=member1
```

Response:

```json
{
  "clusterName": "member1",
  "healthScore": 87.5,
  "reason": "Latency:30.10ms Risk:0.1200 Cost:0.110 -> Score:89.0"
}
```

Examples are provided in `examples/`.
