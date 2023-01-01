from typing import Dict, List, Any

import sys
import base64
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.models.stable_diffusion.constants import _ALPHAS_CUMPROD
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModelV2

class EndpointHandler():
    def __init__(self, path="", version="2"):        
        self.seed = None

        img_height = 512
        img_width = 512
        self.img_height = round(img_height / 128) * 128
        self.img_width = round(img_width / 128) * 128        

        self.MAX_PROMPT_LENGTH = 77

        self.version = version
        self.diffusion_model = self._instantiate_diffusion_model(version)
        if isinstance(self.diffusion_model, str):
          sys.exit(self.diffusion_model)

    def _instantiate_diffusion_model(self, version: str):
        if version == "1.4":
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",
                file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",
            )
            diffusion_model = DiffusionModel(self.img_height, self.img_width, self.MAX_PROMPT_LENGTH)
            diffusion_model.load_weights(diffusion_model_weights_fpath)
            return diffusion_model
        elif version == "2":
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5",
                file_hash="c31730e91111f98fe0e2dbde4475d381b5287ebb9672b1821796146a25c5132d",
            )
            diffusion_model = DiffusionModelV2(self.img_height, self.img_width, self.MAX_PROMPT_LENGTH)
            diffusion_model.load_weights(diffusion_model_weights_fpath)
            return diffusion_model
        else:
            return f"v{version} is not supported"

    def _get_initial_diffusion_noise(self, batch_size, seed):
        if seed is not None:
            return tf.random.stateless_normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4),
                seed=[seed, seed],
            )
        else:
            return tf.random.normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4)
            )

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def __call__(self, data: Dict[str, Any]) -> str:
        # get inputs 
        contexts = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        context = base64.b64decode(contexts[0])
        context = np.frombuffer(context, dtype="float32")
        if self.version == "1.4":
          context = np.reshape(context, (batch_size, 77, 768))
        else:
          context = np.reshape(context, (batch_size, 77, 1024))

        unconditional_context = base64.b64decode(contexts[1])
        unconditional_context = np.frombuffer(unconditional_context, dtype="float32")
        if self.version == "1.4":
          unconditional_context = np.reshape(unconditional_context, (batch_size, 77, 768))
        else:
          unconditional_context = np.reshape(unconditional_context, (batch_size, 77, 1024))

        num_steps = data.pop("num_steps", 25)
        unconditional_guidance_scale = data.pop("unconditional_guidance_scale", 7.5)

        latent = self._get_initial_diffusion_noise(batch_size, self.seed)

        # Iterative reverse diffusion stage
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, unconditional_context]
            )
            latent = self.diffusion_model.predict_on_batch([latent, t_emb, context])
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
            latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            iteration += 1
            progbar.update(iteration)

        latent_b64 = base64.b64encode(latent.numpy().tobytes())
        latent_b64str = latent_b64.decode()

        return latent_b64str
