import json
import base64

from fastapi import FastAPI, File, Form, HTTPException

import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.generative.stable_diffusion.constants import _UNCONDITIONAL_TOKENS

import utils

app = FastAPI()

@app.on_event("startup")
def load_modules():
  version = os.getenv('SD_VERSION', "2")

  global tokenizer
  global text_encoder

  tokenizer = SimpleTokenizer()
  text_encoder = utils.instantiate_text_encoder(version)

  if isinstance(text_encoder, str):
    sys.exit(text_encoder)

@app.post("/text/encode")
async def text_encode(
  prompt: str = "photograph of an astronaut riding a horse", 
  batch_size: int = 1):
  
  encoded_text = utils.encode_text(prompt, tokenizer, text_encoder)
  context, unconditional_context = utils.get_contexts(text_encoder, encoded_text, batch_size)

  context_b64 = base64.b64encode(context.numpy().tobytes())
  context_b64str = context_b64.decode()

  unconditional_context_b64 = base64.b64encode(unconditional_context.numpy().tobytes())
  unconditional_context_b64str = unconditional_context_b64.decode()      

  return {"context": context_b64str, 
          "unconditional_context": unconditional_context_b64str}
