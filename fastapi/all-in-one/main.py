import os
import sys
import json
import base64
import logging
import urllib.request

import keras_cv
from fastapi import FastAPI, File, Form, HTTPException

app = FastAPI()

def _instantiate_stable_diffusion(version):
 if version is "1.4":
  return keras_cv.models.StableDiffusion(img_width=512, img_height=512)
 elif version is "2":
  return keras_cv.models.StableDiffusionV2(img_width=512, img_height=512)
 else:
  return f"v{version} is not supported"
   
@app.on_event("startup")
def load_modules():
 version = os.getenv('SD_VERSION', "2") 
 
 global stable_diffusion
 stable_diffusion = _instantiate_stable_diffusion(version)
 
 if isinstance(self.sd, str):
  sys.exit(self.sd)
 else:
  stable_diffusion.text_to_image("test prompt", batch_size=1)
  logging.warning(f"Stable Diffusion v{version} is fully loaded") 

@app.post("/image/generate")
async def image_generate(
  prompt: str = "photograph of an astronaut riding a horse", 
  batch_size: int = 1):
  
  images = stable_diffusion.text_to_image(prompt, batch_size=batch_size)
  return {"images": base64.b64encode(images.tobytes()).decode()}
