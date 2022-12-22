from typing import Dict, List, Any

import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.decoder import Decoder

img_height = round(512 / 128) * 128
img_width = round(512 / 128) * 128

def decode(decoder, latent):
    decoded = decoder.predict_on_batch(latent)
    decoded = ((decoded + 1) / 2) * 255
    images = np.clip(decoded, 0, 255).astype("uint8")

    return images