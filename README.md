# Various ways of serving Stable Diffusion 

This repository shows a various ways to deploy Stable Diffusion. Currently, we are interested in the Stable Diffusion implementation from `keras-cv`. 

| Title      | Description | Image         | Notebook
| :---:       |    :----   |    :----:     |       :-: |
| All-in-one Stable Diffusion     | This shows how to deploy SD on Hugging Face Endpoint. In this tutorial, you could learn how to base64 encode/decode a batch of images from server to client.       | ![](https://i.ibb.co/0Kpnn8g/2022-12-19-2-57-28.png)   | [Link](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_single_endpoint.ipynb)       |
| Stable Diffusion Separation  | This doesn't show the deployment itself, but you could get some ideas how to separate SD into three part(encoder, diffusion model, decoder) in modular way.        | N/A     |  [Link](https://github.com/deep-diver/keras-sd-serving/blob/main/model_sepration_without_endpoint.ipynb)          |
| Stable Diffusion with Three Endpoints | This is written based on the **Model Separation** notebook. This notebook shows how to deploy each part into separate Hugging Face Endpoints, generate images by interacting each Endpoints, and display the generated images. | N/S | [Link](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_multiple_endpoints.ipynb) |
| Stable Diffusion with One Endpoint(Diffusion Model) | This is written based on the **Stable Diffusion with Three Endpoints** notebook. This notebook shows how to deploy Diffusion Model into Hugging Face Endpoint while having Encoder/Decoder in a local environment. So, you will see how to separate each parts into different environment. | N/A | [Link](https://github.com/deep-diver/keras-sd-serving/blob/main/hf_endpoint_dm_while_local_ed.ipynb) | 
 
