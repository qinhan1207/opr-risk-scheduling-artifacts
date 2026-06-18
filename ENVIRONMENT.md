# Environment Notes

This document summarizes the software environment used or recommended for
inspecting and reproducing the released artifacts. The repository contains
research artifacts rather than a single turnkey deployment package, so exact
cluster manifests and private testbed wiring are not included.

## Operating System

The experiments were prepared and organized on a Windows workstation, while
the Kubernetes/Karmada testbed components run in Linux-based container and VM
environments.

Recommended local inspection environment:

```text
Windows 10/11, Linux, or WSL2
Git
Python 3.10 or newer
```

## Python Environment

The Python code covers two use cases:

- offline data construction and model training in `training_scripts/`
- online OPR-TSMixer inference in `inference_service/`

Recommended Python packages:

```text
python >= 3.10
numpy
pandas
scikit-learn
torch
flask
```

The online inference service requires:

```bash
pip install -r inference_service/requirements.txt
```

The training scripts additionally use `pandas` and `scikit-learn`.

GPU is optional. The released scripts fall back to CPU when CUDA is not
available.

## Model Artifacts

The released online model is:

```text
models/opr-risk/model.pt
```

It is loaded by `inference_service/inference_server_opr.py` together with:

```text
models/opr-risk/scaler.json
models/opr-risk/train_meta.json
```

The model expects an OPR sequence with shape:

```text
10 x 13
```

The implementation-level feature order is documented in
`docs/opr_feature_mapping.md`.

## Java Environment

The Java-side scheduler services are selected integration files from a
multi-module Spring Boot project.

Recommended Java build environment:

```text
JDK 17
Maven 3.9.x
Spring Boot 3.5.x
```

The source modules use Java 17 in their Maven configuration. Maven with a newer
JDK can also compile the selected project, but Java 17 is the intended baseline.

## Go and Karmada Environment

The Go-side scheduler extension is provided as selected Karmada scheduler
plugin files in:

```text
scheduler-plugin/karmada-go-extension/
```

Source context:

```text
Karmada: v1.15.0-oriented development tree
Kubernetes: v1.31.14 in the experimental testbed
Go: 1.24.x was used in the local development environment
```

The plugin is named:

```text
GlobalAdvisor
```

It calls the Java global scheduler endpoint:

```text
GET /api/advisor/score?cluster=<clusterName>
```

The Java service then calls the Python OPR-TSMixer inference service.

## Testbed Software

The controlled multi-cluster testbed uses:

```text
Kubernetes v1.31.14
Karmada v1.15.0
kind-based member clusters
k6 workload replay
tc/NetEm controlled network perturbation
```

The member-cluster roles and network settings are documented in
`configs/README.md`.

## Reproducibility Notes

- The released data files are collected observations from the controlled
  testbed.
- Re-training can produce small numerical differences because of hardware,
  library versions, and random initialization.
- The selected Java and Go files document the scheduler integration logic, but
  deployment-specific manifests, credentials, and private environment wiring
  are intentionally excluded.
