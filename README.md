# Various ways of serving Stable Diffusion 

This repository shows a various ways to deploy Stable Diffusion. Currently, we are interested in the Stable Diffusion implementation from `keras-cv`, and the target platforms/frameworks that we aim includes [TF Serving](https://github.com/tensorflow/serving), [Hugging Face Endpoint](https://huggingface.co/inference-endpoints), and [FastAPI](https://fastapi.tiangolo.com/). 

From the version `0.4.0` release of `keras-cv`, `StableDiffusionV2` is included, and this repository support both version 1 and 2 of the Stable Diffusion. 

## 1. All in One Endpoint

This method shows how to deploy Stable Diffusion as a whole in a single endpoint. Stable Diffusion consists of three models(`encoder`, `diffusion model`, `decoder`) and some glue codes to handle the inputs and outputs of each models. In this scenario, everything is packaged into a single Endpoint.

<p align="center">
<img src="https://github.com/deep-diver/keras-sd-serving/blob/main/assets/all-in-one.png?raw=true" width="90%"/>
</p>

- **Hugging Face 🤗 Endpoint**: In order to deploy something in Hugging Face Endpoint, we need to create a [custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler). Hugging Face Endpoint let us easily deploy any machine learning models with pre/post processing logics in a custom handler [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/hfe_all_in_one.ipynb) | [Standalone Codebase](https://github.com/deep-diver/keras-sd-serving/tree/main/hf_custom_handlers/all-in-one)]

- **FastAPI Endpoint**: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/fastapi_all_in_one.ipynb) | [Standalone](https://github.com/deep-diver/keras-sd-serving/blob/main/fastapi/basic-diffusion/utils.py)]
  - Docker Image: `gcr.io/gcp-ml-172005/sd-fastapi-allinone:latest`

## 2. Three Endpoints 

This method shows how to deploy Stable Diffusion in three separate Endpoints. As a preliminary work, [this notebook](https://github.com/deep-diver/keras-sd-serving/blob/main/model_sepration_without_endpoint.ipynb) was written to demonstrate how to split three parts of Stable Diffusion into three separate modules. In this example, you will see how to interact with three different endpoints to generate images with a given text prompt.

<p align="center">
<img src="https://github.com/deep-diver/keras-sd-serving/blob/main/assets/three-endpoints.png?raw=true" width="90%"/>
</p>

- **Hugging Face Endpoint**: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/hfe_three_endpoints.ipynb) | [Text Encoder](https://github.com/deep-diver/keras-sd-serving/tree/main/hf_custom_handlers/text-encoder) | [Diffusion Model](https://github.com/deep-diver/keras-sd-serving/tree/main/hf_custom_handlers/basic-diffusion) | [Decoder](https://github.com/deep-diver/keras-sd-serving/tree/main/hf_custom_handlers/decoder)]

- **FastAPI Endpoint**: [[Central](https://github.com/deep-diver/keras-sd-serving/tree/main/fastapi/central) | [Text Encoder](https://github.com/deep-diver/keras-sd-serving/tree/main/fastapi/text-encoder) | [Diffusion Model](https://github.com/deep-diver/keras-sd-serving/tree/main/fastapi/basic-diffusion) | [Decoder](https://github.com/deep-diver/keras-sd-serving/tree/main/fastapi/decoder)]
  - Docker Image(text-encoder): `gcr.io/gcp-ml-172005/sd-fastapi-text-encoder:latest`
  - Docker Image(diffusion-model): `gcr.io/gcp-ml-172005/sd-fastapi-diffusion-model:latest`
  - Docker Image(decoder): `gcr.io/gcp-ml-172005/sd-fastapi-decoder:latest`

- **TF Serving Endpoint**: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/tfs_three_endpoints.ipynb) | [Dockerfiles + k8s Resources](https://github.com/deep-diver/keras-sd-serving/tree/main/tfserving)]
  - SavedModel: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/tfs_saved_models.ipynb) | [Text Encoder](https://huggingface.co/keras-sd/tfs-text-encoder/tree/main) | [Diffusion Model](https://huggingface.co/keras-sd/tfs-diffusion-model/tree/main) | [Decoder](https://huggingface.co/keras-sd/tfs-decoder/tree/main)]
    - wrapping `encoder`, `diffusion model`, and `decoder` and some glue codes in separate [SavedModel](https://www.tensorflow.org/guide/saved_model)s. With them, we can not only deploy each models on cloud with TF Serving but also embed in web and mobild applications with [TFJS](https://github.com/tensorflow/tfjs) and [TFLite](https://www.tensorflow.org/lite). We will explore the embedded use cases later phase of this project.
  - Docker Images
    - text-encoder: `gcr.io/gcp-ml-172005/tfs-sd-text-encoder:latest`
    - text-encoder w/ base64: `gcr.io/gcp-ml-172005/tfs-sd-text-encoder-base64:latest`
    - text-encoder-v2: `gcr.io/gcp-ml-172005/tfs-sd-text-encoder-v2:latest`
    - text-encoder-v2 w/ base64: `gcr.io/gcp-ml-172005/tfs-sd-text-encoder-v2-base64:latest`
    - diffusion-model: `gcr.io/gcp-ml-172005/tfs-sd-diffusion-model:latest`
    - diffusion-model w/ base64: `gcr.io/gcp-ml-172005/tfs-sd-diffusion-model-base64:latest`
    - diffusion-model-v2: `gcr.io/gcp-ml-172005/tfs-sd-diffusion-model-v2:latest`
    - diffusion-model-v2 w/ base64: `gcr.io/gcp-ml-172005/tfs-sd-diffusion-model-v2-base64:latest`
    - decoder: `gcr.io/gcp-ml-172005/tfs-sd-decoder:latest`  
    - decoder w/ base64: `gcr.io/gcp-ml-172005/tfs-sd-decoder-base64:latest`
  
> NOTE: Passing intermediate values between models through network could be costly, and some platform limits certain payload size. For instance, [Vertex AI limits the request size to 1.5MB](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions#send_an_online_prediction_request). To this end, we provide different TF Serving Docker images which handles inputs and produces outputs in `base64` format.


## 3. One Endpoint with Two local APIs (w/ 🤗 Endpoint) 

With the separation of Stable Diffusion, we could organize each parts in any environments. This is powerful especially if we want to deploy specialized `diffusion model`s such as `inpainting` and `finetuned diffusion model`. In this case, we only need to replace the currently deployed `diffusion model` or just deploy a new `diffusion model` besides while keeping the other two(`text encoder` and `decoder`) as is.

Also, it is worth noting that we could run `text encoder` and `decoder` parts in local(Python clients or web/mobile with TF Serving) while having `diffusion model` on cloud. In this repository, we currently show an example using Hugging Face 🤗 Endpoint. However, you could easily expand the posibilities.

> NOTE: along with this project, we have developed one more project to fine-tune Keras based Stable Diffusion at [**Fine-tuning Stable Diffusion using Keras**](https://github.com/sayakpaul/stable-diffusion-keras-ft). We currently provide a fine-tuned model to Pokemon dataset. 

<p align="center">
<img src="https://github.com/deep-diver/keras-sd-serving/blob/main/assets/one-endpoint-two-local.png?raw=true" width="90%"/>
</p>

- **Original txt2img generation**: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/notebooks/hfe_two_endpoints_one_local_diffusion.ipynb)]

- **Original inpainting**: [[Colab](https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/hfe_two_endpoints_one_local_inpainting.ipynb)]

## 4. On-Device Deployment (w/ TFLite) - WIP

We have managed to convert `SavedModel`s into TFLite models, and we are hosting them as below (thanks to @farmaker47):
- [Text Encoder TFLite Model](https://huggingface.co/keras-sd/text-encoder-tflite) - 127MB
- [Diffusion Model TFLite Model](https://huggingface.co/keras-sd/diffusion-model-tflite) - 1.7GB
- [Decoder TFLite Model](https://huggingface.co/keras-sd/decoder-tflite) - 99MB

These TFLite models have the same signature as the `SavedModel`s, and all the pre/post operations are included inside. All of them are converted with float16 quantization optimize process. You can find more about how to convert `SavedModel`s to `TFLite` models in this [repository](https://github.com/farmaker47/diffusion_models_tflite_conversion_and_inference).

TODO
- [ ] Implement SimpleTokenizer in JAVA and JavaScript
- [ ] Run TFLite models on Android and Web browser 

## Timing Tests

<details><summary>details</summary>
<p>

### Sequential

The figure below shows how long each scenario took from text encoding to diffusion to decoding. It assumes each request(`batch_size=4`) is handled sequentially with a single server running on Hugging Face Endpoint for each endpoint. **all-in-one endpoint** deployed the Stable Diffusion on A10 equipped server while **separate endpoints** deployed text encoder on 2 vCPU + 4GB RAM, diffusion model on A10 equipped server, and decoder on T4 equipped server. Finally, **one endpoint, two local** only deployed difusion model on A10 equipped server while keeping the other two on Colab environment (w/ T4). Please take a look how these are measured from [this notebook](https://github.com/deep-diver/keras-sd-serving/blob/main/timings_sequential.ipynb)

![](https://i.ibb.co/PQX9xt5/download-1.png)

</p>
</details>

## Acknowledgements
Thanks to the ML Developer Programs' team at Google for providing GCP credits.
