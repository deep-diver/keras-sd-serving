# Various ways of serving Stable Diffusion 

This repository shows a various ways to deploy Stable Diffusion. Currently, we are interested in the Stable Diffusion implementation from `keras-cv`. 

- [`hf_single_endpoint.ipynb`](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_single_endpoint.ipynb): This shows how to deploy SD on Hugging Face Endpoint. In this tutorial, you could learn how to base64 encode/decode a batch of images from server to client. 

- [`model_sepration_without_endpoint.ipynb`](https://github.com/deep-diver/keras-sd-serving/blob/main/model_sepration_without_endpoint.ipynb): This doesn't show the deployment itself, but you could get some ideas how to separate SD into three part(encoder, diffusion model, decoder) in modular way.

- [`hf_multiple_endpoints.ipynb`](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_multiple_endpoints.ipynb): This is written based on the `model_sepration_without_endpoint.ipynb` notebook. This notebook shows how to deploy each part into separate Hugging Face Endpoints, generate images by interacting each Endpoints, and display the generated images.

- [`hf_endpoint_dm_while_local_ed.ipynb`](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_endpoint_dm_while_local_ed.ipynb): This is written based on the `hf_multiple_endpoints.ipynb` notebook. This notebook shows how to deploy Diffusion Model into Hugging Face Endpoint while having Encoder/Decoder in a local environment. So, you will see how to separate each parts into different environment.
