from typing import Dict, List, Any
import base64

import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.text_encoder import TextEncoder
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.generative.stable_diffusion.constants import _UNCONDITIONAL_TOKENS

class EndpointHandler():
    def __init__(self, path=""):
        self.MAX_PROMPT_LENGTH = 77

        self.tokenizer = SimpleTokenizer()
        self.text_encoder = TextEncoder(self.MAX_PROMPT_LENGTH)
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
            file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
        )
        self.text_encoder.load_weights(text_encoder_weights_fpath)        
        self.pos_ids = tf.convert_to_tensor([list(range(self.MAX_PROMPT_LENGTH))], dtype=tf.int32)    

    def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, self.pos_ids]
        )

        return unconditional_context

    def encode_text(self, prompt):
      # Tokenize prompt (i.e. starting context)
      inputs = self.tokenizer.encode(prompt)
      if len(inputs) > self.MAX_PROMPT_LENGTH:
          raise ValueError(
              f"Prompt is too long (should be <= {self.MAX_PROMPT_LENGTH} tokens)"
          )
      phrase = inputs + [49407] * (self.MAX_PROMPT_LENGTH - len(inputs))
      phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

      context = self.text_encoder.predict_on_batch([phrase, self.pos_ids])

      return context  

    def get_contexts(self, encoded_text, batch_size):
        encoded_text = tf.squeeze(encoded_text)
        if encoded_text.shape.rank == 2:
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), batch_size, axis=0
            )

        context = encoded_text

        unconditional_context = tf.repeat(
            self._get_unconditional_context(), batch_size, axis=0
        )  

        return context, unconditional_context

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # get inputs 
        prompt = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        encoded_text = self.encode_text(prompt)
        context, unconditional_context = self.get_contexts(encoded_text, batch_size)

        context_b64 = base64.b64encode(context.numpy().tobytes())
        context_b64str = context_b64.decode()

        unconditional_context_b64 = base64.b64encode(unconditional_context.numpy().tobytes())
        unconditional_context_b64str = unconditional_context_b64.decode()        
        
        return {"context_b64str": context_b64str, "unconditional_context_b64str": unconditional_context_b64str}