from typing import Dict, List, Any

import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.decoder import Decoder

class EndpointHandler():
    def __init__(self, path=""):        
        img_height = 512
        img_width = 512
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        
        self.decoder = Decoder(img_height, img_width)
        decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
            file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
        )
        self.decoder.load_weights(decoder_weights_fpath)

    def __call__(self, data: Dict[str, Any]) -> str:
        # get inputs 
        latent = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        latent = base64.b64decode(latent)
        latent = np.frombuffer(latent, dtype="float32")
        latent = np.reshape(latent, (batch_size, 64, 64, 4))

        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        images = np.clip(decoded, 0, 255).astype("uint8")

        images_b64 = base64.b64encode(images.tobytes())
        images_b64str = images_b64.decode()

        return images_b64str             