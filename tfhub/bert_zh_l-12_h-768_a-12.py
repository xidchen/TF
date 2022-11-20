"""
https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4

BERT provides dense vector representations for natural language by using
a deep, pre-trained neural network with the Transformer architecture.

The weights of this model are those released by the original BERT authors.
This model has been pre-trained for Chinese on the Wikipedia. For training,
random input masking has been applied independently to word pieces
(as in the original BERT paper).

All parameters in the module are trainable, and fine-tuning all parameters
is the recommended practice.

This SavedModel implements the encoder API for text embeddings with
transformer encoders. It expects a dict with three int32 Tensors as input:
input_word_ids, input_mask, and input_type_ids.

The separate preprocessor SavedModel at
https://tfhub.dev/tensorflow/bert_zh_preprocess/3 transforms plain text inputs
into this format, which its documentation describes in greater detail.
"""

import tensorflow as tf
import tensorflow_hub as hub


def basic_usage():
    """
    The simplest way to use this model in the Keras functional API.
    The encoder's output can be pooled_output and sequence_output,
    pooled_output to represent each input sequence as a whole,
    sequence_output to represent each input token in context.
    Either of those can be used as input to further model building.
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_zh_preprocess/3")
    encoder_inputs = preprocessor(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4",
        trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]
    embedding_model = tf.keras.Model(text_input, pooled_output)
    sentences = tf.constant(["(your text here)"])
    print(embedding_model(sentences))


def advanced_topics(seq_length: int):
    """
    The preprocessor documentation explains how to input segment pairs and how
    to control seq_length.
    The intermediate activations of all L=12 Transformer blocks (hidden layers)
    are returned as a Python list: outputs["encoder_outputs"][i] is a Tensor
    of shape [batch_size, seq_length, 768] with the outputs of the i-th
    Transformer block, for 0 <= i < L. The last value of the list is equal
    to sequence_output.
    The preprocessor can be run from inside a callable passed
    to tf.data.Dataset.map() while this encoder stays a part of a larger model
    that gets trained on that dataset.
    """
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    )
    return encoder_inputs


def masked_language_model(seq_length: int, num_predict: int):
    """
    This SavedModel provides a trainable .mlm sub-object with predictions
    for the Masked Language Model task it was originally trained with.
    This allows advanced users to continue MLM training for fine-tuning
    to a downstream task. It extends the encoder interface above
    with a zero-padded tensor of positions in the input sequence
    for which the input_word_ids have been randomly masked or altered.
    """
    mlm_inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        masked_lm_positions=tf.keras.layers.Input(shape=(num_predict,), dtype=tf.int32),
    )

    encoder = hub.load("https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/4")
    mlm = hub.KerasLayer(encoder.mlm, trainable=True)
    mlm_outputs = mlm(mlm_inputs)
    return mlm_outputs["mlm_logits"]
