We are considering GKE(Google Kubernetes Engine) in particular to deploy FastAPI endpoints of Stable Diffusion.

## Setup

In order to deploy Stable Diffusion, we almost always need GPUs. There are three steps to enable GPUs on GKE. For further detail, please take a look at this [official documentation about GPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus).

### GKE Cluster

```
gcloud container clusters create v100 \
  --accelerator type=nvidia-tesla-v100,count=2 \
  --zone us-central1-c
```

### Dockerfile

```
FROM tensorflow/tensorflow:latest-gpu

...
```

### Deployment.yaml

```
apiVersion: apps/v1
kind: Deployment
...
    spec:
      containers:
...
        resources:
          limits:
            nvidia.com/gpu: 1
```
