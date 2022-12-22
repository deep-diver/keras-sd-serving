import base64
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.constants import _ALPHAS_CUMPROD

seed = None

MAX_PROMPT_LENGTH = 77

img_height = round(512 / 128) * 128
img_width = round(512 / 128) * 128     

def get_initial_diffusion_noise(batch_size):
  if seed is not None:
      return tf.random.stateless_normal(
          (batch_size, img_height // 8, img_width // 8, 4),
          seed=[seed, seed],
      )
  else:
      return tf.random.normal(
          (batch_size, img_height // 8, img_width // 8, 4)
      )

def get_initial_alphas(timesteps):
  alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
  alphas_prev = [1.0] + alphas[:-1]

  return alphas, alphas_prev

def get_timestep_embedding(timestep, batch_size, dim=320, max_period=10000):
  half = dim // 2
  freqs = tf.math.exp(
      -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
  )
  args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
  embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
  embedding = tf.reshape(embedding, [1, -1])
  return tf.repeat(embedding, batch_size, axis=0)

def diffusion(
  context,
  unconditional_context,
  diffusion_model, 
  num_steps, 
  unconditional_guidance_scale, 
  batch_size):

  latent = get_initial_diffusion_noise(batch_size, seed)

  # Iterative reverse diffusion stage
  timesteps = tf.range(1, 1000, 1000 // num_steps)
  alphas, alphas_prev = get_initial_alphas(timesteps)
  progbar = keras.utils.Progbar(len(timesteps))
  iteration = 0
  for index, timestep in list(enumerate(timesteps))[::-1]:
      latent_prev = latent  # Set aside the previous latent vector
      t_emb = get_timestep_embedding(timestep, batch_size)
      unconditional_latent = diffusion_model.predict_on_batch(
          [latent, t_emb, unconditional_context]
      )
      latent = diffusion_model.predict_on_batch([latent, t_emb, context])
      latent = unconditional_latent + unconditional_guidance_scale * (
          latent - unconditional_latent
      )
      a_t, a_prev = alphas[index], alphas_prev[index]
      pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
      latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
      iteration += 1
      progbar.update(iteration)

  return latent
