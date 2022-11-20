"""
https://tfhub.dev/tensorflow/bert_zh_preprocess/3

Text preprocessing for BERT.

This SavedModel is a companion of BERT models to preprocess plain text inputs
into the input format expected by BERT. Check the model documentation
to find the correct preprocessing model for each particular BERT
or other Transformer encoder model.

This model uses a vocabulary for Chinese extracted from the Wikipedia
(same as in the models by the original BERT authors).

This model has no trainable parameters and can be used in an input pipeline
outside the training loop.

This SavedModel implements the preprocessor API for text embeddings
with Transformer encoders, which offers several ways to go from one
or more batches of text segments (plain text encoded as UTF-8) to the inputs
for the Transformer encoder model.
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text


def basic_usage() -> tf.Tensor:
    """
    Basic usage for single segments.
    Inputs with a single text segment can be mapped to encoder inputs like this.
    :return: encoder inputs, whose seq_length=128.
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_zh_preprocess/3")
    return preprocessor(text_input)


def general_usage() -> tf.Tensor:
    """
    General usage.
    For pairs of input segments, to control the seq_length,
    or to modify tokenized sequences before packing them into encoder inputs,
    the preprocessor can be called like this.

    The call to tokenize() returns an int32 RaggedTensor
    of shape [batch_size, (words), (tokens_per_word)]. Correspondingly,
    the call to bert_pack_inputs() accepts a RaggedTensor
    of shape [batch_size, ...] with rank 2 or 3.
    :return: encoder inputs, whose seq_length can be customized.
    """
    preprocessor = hub.load(
        "https://tfhub.dev/tensorflow/bert_zh_preprocess/3")

    # Step 1: tokenize batches of text inputs.
    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string),
                   ...]  # This SavedModel accepts up to 2 text inputs.
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]

    # Step 2 (optional): modify tokenized inputs.
    pass

    # Step 3: pack input sequences for the Transformer encoder.
    seq_length = 128  # Your choice here.
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    return bert_pack_inputs(tokenized_inputs)
