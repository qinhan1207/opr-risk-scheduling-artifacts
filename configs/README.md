# Testbed Configuration Notes

This directory documents the experimental environment used for the released
artifacts.

## Software Stack

- Kubernetes: v1.31.14
- Karmada: v1.15.0
- Workload replay: k6
- Member-cluster construction: kind-based Kubernetes clusters

## Cluster Roles

| Cluster | Node layout | Network setting | Scenario role |
| --- | --- | --- | --- |
| member1 | 1 control-plane, 5 workers | 40 ms delay, no jitter | Resource-rich, network-poor |
| member2 | 1 control-plane, 3 workers | 10 ms delay, no jitter | Stable low-latency |
| member3 | 1 control-plane, 3 workers | 18 ms delay, 15 ms jitter | Volatile, misleading snapshot |
| member4 | 1 control-plane, 2 workers | 15 ms delay, no jitter | Stable, resource-weaker |
| member5 | 1 control-plane, 3 workers | 15 ms delay, no jitter | Stable, secondary candidate |

## Network Perturbation

The controlled perturbation utility is provided at
`scripts/auto_fault_injector03.sh`. It applies repeatable latency-risk patterns
to the testbed so that stable, onset, persistent, and recovery states can be
observed in the collected traces.

## Notes

The exact Kubernetes and Karmada deployment manifests are environment-specific.
The released data and result summaries are intended to support inspection of the
reported experiments without requiring the complete private testbed deployment.

