# Various ways of Serving Stable Diffusion 

This repository shows a various ways to deploy Stable Diffusion. Currently, we are interested in the Stable Diffusion implementation from `keras-cv`. 

- `hf_single_endpoint.ipynb`: This shows how to deploy SD on Hugging Face Endpoint. In this tutorial, you could learn how to base64 encode/decode a batch of images from server to client. 
- `model_sepration_without_endpoint.ipynb`: This doesn't show the deployment itself, but you could get some ideas how to separate SD into three part(encoder, diffusion model, decoder) in modular way.
- `hf_multiple_endpoints.ipynb`: This is written based on the `model_sepration_without_endpoint.ipynb` notebook. This notebook shows how to deploy each part into separate Hugging Face Endpoints, generate images by interacting each Endpoints, and display the generated images.
