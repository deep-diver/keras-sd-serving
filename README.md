# Various ways of serving Stable Diffusion 

This repository shows a various ways to deploy Stable Diffusion. Currently, we are interested in the Stable Diffusion implementation from `keras-cv`. The codes inside every notebooks are written based on `keras-cv == 0.3.4`.

> NOTE: `keras-cv >= 0.3.5` is released, but it was not registered in [PyPI](https://pypi.org/project/keras-cv/) at the time of creating this repository. When it is distributed to PyPI, the notebooks will be updated accordingly.

## 1. All in One Endpoint (w/ 🤗 Endpoint) 

<a target="_blank" href="https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/hfe_all_in_one.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This method shows how to create a [custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler) of Hugging Face 🤗 Endpoint. Stable Diffusion consists of three models(`encoder`, `diffusion model`, `decoder`) and some glue codes to handle the inputs and outputs of each models. In this scenario, everything is packaged into a single Endpoint. Hugging Face 🤗 Endpoint let us easily deploy any machine learning models with pre/post processing logics in custom handler.

<details><summary>details</summary>
<p>

<p align="center">
<img src="https://i.ibb.co/0Kpnn8g/2022-12-19-2-57-28.png" width="70%"/>
</p>

</p>
</details>

## 2. Three Endpoints (w/ 🤗 Endpoint) 

<a target="_blank" href="https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/hfe_three_endpoints.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This method shows how to create three [custom handler](https://huggingface.co/docs/inference-endpoints/guides/custom_handler)s of Hugging Face 🤗 Endpoint. As a preliminary work, [this notebook](https://github.com/deep-diver/keras-sd-serving/blob/main/model_sepration_without_endpoint.ipynb) was written first to demonstrate how to split three parts of Stable Diffusion into separate modules. In this example, you will see how to interact with three different endpoints to generate images with a given text prompt.

<details><summary>details</summary>
<p>

<p align="center">
<img src="https://i.ibb.co/1dCGfm9/2022-12-19-3-27-14.png" width="70%"/>
</p>

</p>
</details>

## 3. One Endpoint with Two APIs on local for txt2img (w/ 🤗 Endpoint) 

<a target="_blank" href="https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/hfe_two_endpoints_one_local_diffusion.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This method shows how to generate images with a given text prompt by interacting with three parts of Stable Diffusion. There is one significat difference comparing to **2. Three Endpoints (w/ 🤗 Endpoint)**. In this scenario, diffusion model is deployed on the Hugging Face Endpoint while keeping other models in local environment. It basically shows the flexibility of organizing Stable Diffusion in various situations (i.e. `text encoder`: local, `diffusion model`/`decoder`: Cloud, etc.,).

<details><summary>details</summary>
<p>

<p align="center">
<img src="https://i.ibb.co/f2NHXYh/2022-12-19-3-27-10.png" width="70%"/>
</p>

</p>
</details>

## 4. One Endpoint with Two APIs on local for inpainting (w/ 🤗 Endpoint) 

<a target="_blank" href="https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/hfe_two_endpoints_one_local_inpainting.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This method shows how to inpaint a given image with a given text prompt by interacting with three parts of Stable Diffusion. The architecture is bacisally same to **3. One Endpoint with Two APIs on local for txt2img (w/ 🤗 Endpoint)**. However, it significantly demonstrate the flexibility or replacing **only** `diffusion model` with other specialized one. `text encoder` and `decoder` remains the same as is while the basic `diffusion model` for txt2img is only replaced with specialied one for inpainting task.

<details><summary>details</summary>
<p>

<p align="center">
<img src="https://i.ibb.co/fv30h2M/2022-12-20-3-17-57.png" width="70%"/>
</p>

</p>
</details>

## 5. Three Endpoints (w/ TF Serving)

<a target="_blank" href="https://colab.research.google.com/github/deep-diver/keras-sd-serving/blob/main/tfs_three_endpoints.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This method shows how to wrap `encoder`, `diffusion model`, and `decoder` in separate [TF Serving](https://github.com/tensorflow/serving). TF Serving is a specialized ML deployment framework, so there are many benefits you could get out of the box such as batch prediction. Also, each part should be saved as `SavedModel` format, so this method will be a stepping stone to deploy each parts into different deployment targets such as Mobile, Web, and more. 

## Timing Tests

### Sequential

The figure below shows how long each scenario took from text encoding to diffusion to decoding. It assumes each request(`batch_size=4`) is handled sequentially with a single server running on Hugging Face Endpoint for each endpoint. **all-in-one endpoint** deployed the Stable Diffusion on A10 equipped server while **separate endpoints** deployed text encoder on 2 vCPU + 4GB RAM, diffusion model on A10 equipped server, and decoder on T4 equipped server. Finally, **one endpoint, two local** only deployed difusion model on A10 equipped server while keeping the other two on Colab environment (w/ T4). Please take a look how these are measured from [this notebook](https://github.com/deep-diver/keras-sd-serving/blob/main/timings_sequential.ipynb)

![](https://i.ibb.co/PQX9xt5/download-1.png)
