We are considering GKE(Google Kubernetes Engine) in particular to deploy FastAPI endpoints of Stable Diffusion.

## Setup

In order to deploy Stable Diffusion, we almost always need GPUs. There are three steps to enable GPUs on GKE. For further detail, please take a look at this [official documentation about GPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus).

### GKE Cluster

When creating a GKE cluster, you need to specify `accelerator` option with desired GPU type and the number of GPUs to be attatched to each node.

```
gcloud container clusters create {CLUSTER_NAME} \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --zone={CLUSTER_ZONE} \
  --num-nodes=3 \ 
  --min-nodes=3
```

### Install NVIDIA GPU device drivers

Creating a GKE cluster with accelerator attatched doesn't mean the system knows how to interact with GPUs. You should install NVIDIA GPU device drivers with the following command which runs special pods for that purpose.

```
$ kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Dockerfile and deployment.yaml

Lastly, you need to build a Docker image that is installed NVIDIA GPU drivers for Docker container. There is a general process to do this, but we have simply used Tensorflow official Docker image with GPU enabled.

```
FROM tensorflow/tensorflow:latest-gpu

...
```

The deployment.yaml also should have special value of `nvidia.com/gpu` to seek GPU equipped node to deploy the Deployment.

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

## Provision k8s resources

After everything is set up, you could simply create/provision k8s resources(`deployment` and `service`). To do this, you need to connect to the cluster that you want to provision the resource. The connection CLI should be similar to the below:

```
$ gcloud container clusters get-credentials {CLUSTER_NAME} --zone {CLUSTER_ZONE} --project {PROJECT_ID}
```

After you connect to the cluster successfully, then simply run `kubectl apply` CLI like below:

```
# in the case of all-in-one case

$ kubectl apply -f all-in-one/deployment.yaml
$ kubectl apply -f all-in-one/service.yaml
```
