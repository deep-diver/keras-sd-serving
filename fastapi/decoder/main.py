import json
import base64

from fastapi import FastAPI, File, Form, HTTPException

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.decoder import Decoder

import utils

app = FastAPI()

@app.on_event("startup")
def load_modules():
	global decoder

	decoder = Decoder(utils.img_height, utils.img_width)
	decoder_weights_fpath = keras.utils.get_file(
		origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
		file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
	)
	decoder.load_weights(decoder_weights_fpath)

@app.post("/decode")
async def decode(
	latent: str = None,
	batch_size: int = None):

	if latent is None:
		return "latent is not passed"

	if batch_size is None:
		return "batch_size is not passed"

	latent = base64.b64decode(latent)
	latent = np.frombuffer(latent, dtype="float32")
	latent = np.reshape(latent, (batch_size, 64, 64, 4))

	images = utils.decode(decoder, latent)
	images_b64 = base64.b64encode(images.tobytes())
	images_b64str = images_b64.decode()
	
	return {"images": images_b64str}
