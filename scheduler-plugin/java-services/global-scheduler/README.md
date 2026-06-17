# Global Scheduler

The global scheduler receives observations from Local Scheduler Agents,
maintains the latest member-cluster status, calls the Python OPR risk predictor,
and exposes scheduler-consumable scores to the Karmada-side Go extension.

## Data Flow

1. `POST /api/clusters/report`
   - Receives `ClusterStatus` from each Local Scheduler Agent.
   - The payload contains current aggregated latency/loss, raw peer probe
     results, and a recent `latencyWindow`.

2. `MemberClusterServiceImpl`
   - Optionally performs cloud-side aggregation for backward compatibility.
   - Calls `NetworkStabilityServiceImpl` to evaluate placement risk.
   - Stores the latest status in memory.

3. `NetworkStabilityServiceImpl`
   - Sends the recent `latencyWindow` to the Python OPR predictor.
   - Receives a short-term placement-risk probability.
   - Fuses current network cost and predicted risk into `stabilityScore`.

4. `GET /api/advisor/score?cluster=<name>`
   - Returns the risk-aware `ClusterScore` consumed by the scheduler.

5. `GET /api/advisor/network-score?cluster=<name>`
   - Returns a network-only score for ablation and comparison experiments.

## Important Fields

- `networkLatency`: current member-cluster-level latency observation.
- `packetLossRate`: current aggregated packet loss rate.
- `latencyWindow`: recent latency observations used by Python to construct OPR
  features.
- `stabilityScore`: final scheduler-side score computed from network cost and
  OPR risk probability.
- `finalScore`: score returned to the Karmada-side scheduler extension.

