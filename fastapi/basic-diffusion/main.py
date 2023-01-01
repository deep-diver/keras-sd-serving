import os
import sys
import json
import base64

from fastapi import FastAPI, File, Form, HTTPException

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.diffusion_model import DiffusionModel

import utils

version = os.getenv('SD_VERSION', "2")

app = FastAPI()

@app.on_event("startup")
def load_modules():
  global diffusion_model

  diffusion_model = instantiate_diffusion_model(version)

  if isinstance(diffusion_model, str):
    sys.exit(diffusion_model)

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
  if version == "2":
    context = np.reshape(context, (batch_size, 77, 1024))
  else:
    context = np.reshape(context, (batch_size, 77, 768))

  unconditional_context = base64.b64decode(u_context)
  unconditional_context = np.frombuffer(unconditional_context, dtype="float32")
  if version == "2":
    unconditional_context = np.reshape(unconditional_context, (batch_size, 77, 1024))
  else:
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
