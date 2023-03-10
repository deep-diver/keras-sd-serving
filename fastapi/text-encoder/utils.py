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

def instantiate_text_encoder(version: str):
    if version == "1.4":
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
            file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
        )
        text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
        text_encoder.load_weights(text_encoder_weights_fpath)
        return text_encoder
    elif version == "2":
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/text_encoder_v2_1.h5",
            file_hash="985002e68704e1c5c3549de332218e99c5b9b745db7171d5f31fcd9a6089f25b",
        )
        text_encoder = TextEncoderV2(MAX_PROMPT_LENGTH)
        text_encoder.load_weights(text_encoder_weights_fpath)
        return text_encoder
    else:
        return f"v{version} is not supported"