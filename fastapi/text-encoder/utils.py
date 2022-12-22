import tensorflow as tf
from keras_cv.models.generative.stable_diffusion.constants import _UNCONDITIONAL_TOKENS

MAX_PROMPT_LENGTH = 77

pos_ids = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

def get_unconditional_context(text_encoder):
    unconditional_tokens = tf.convert_to_tensor(
        [_UNCONDITIONAL_TOKENS], dtype=tf.int32
    )
    unconditional_context = text_encoder.predict_on_batch(
        [unconditional_tokens, pos_ids]
    )

    return unconditional_context

def encode_text(prompt, tokenizer, text_encoder):
  inputs = tokenizer.encode(prompt)
  phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
  phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

  encoded_text = text_encoder.predict_on_batch([phrase, pos_ids])
  return encoded_text  

def get_contexts(text_encoder, encoded_text, batch_size):
    encoded_text = tf.squeeze(encoded_text)
    if encoded_text.shape.rank == 2:
        encoded_text = tf.repeat(
            tf.expand_dims(encoded_text, axis=0), batch_size, axis=0
        )

    context = encoded_text

    unconditional_context = tf.repeat(
        get_unconditional_context(text_encoder), batch_size, axis=0
    )  

    return context, unconditional_context
