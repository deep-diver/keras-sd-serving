import json
import base64

from fastapi import FastAPI, File, Form, HTTPException

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.diffusion_model import DiffusionModel

import utils

app = FastAPI()

@app.on_event("startup")
def load_modules():
  global diffusion_model

  diffusion_model = DiffusionModel(utils.img_height, 
                                utils.img_width, 
                                utils.MAX_PROMPT_LENGTH)

  diffusion_model_weights_fpath = keras.utils.get_file(
      origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
      file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
  )
  diffusion_model.load_weights(diffusion_model_weights_fpath)                                        

@app.post("/diffusion")
async def diffusion(
  context: str = None, 
  u_context: str = None,
  batch_size: int = None,
  num_steps: int = 25,
  unconditional_guidance_scale: float = 7.5):

  if context is None:
    return "context is not passed"

  if u_context is None:
    return "unconditional_context is not passed"
  
  if batch_size is None:
    return "batch_size is not passed"

  context = base64.b64decode(context)
  context = np.frombuffer(context, dtype="float32")
  context = np.reshape(context, (batch_size, 77, 768))

  unconditional_context = base64.b64decode(u_context)
  unconditional_context = np.frombuffer(unconditional_context, dtype="float32")
  unconditional_context = np.reshape(unconditional_context, (batch_size, 77, 768))

  latent = utils.diffusion(context,
                unconditional_context,
                diffusion_model,
                num_steps,
                unconditional_guidance_scale,
                batch_size)

  latent_b64 = base64.b64encode(latent.numpy().tobytes())
  latent_b64str = latent_b64.decode()

  return {"latent": latent_b64str}
