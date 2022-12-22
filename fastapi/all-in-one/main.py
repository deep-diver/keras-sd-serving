import json
import base64
import urllib.request

import keras_cv
from fastapi import FastAPI, File, Form, HTTPException

app = FastAPI()

@app.on_event("startup")
def load_modules():
 global stable_diffusion
 stable_diffusion = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

@app.post("/image/generate")
async def image_generate(
  prompt: str = "photograph of an astronaut riding a horse", 
  batch_size: int = 1):
  
  images = stable_diffusion.text_to_image(prompt, batch_size=batch_size)
  return {"images": base64.b64encode(images.tobytes()).decode()}
