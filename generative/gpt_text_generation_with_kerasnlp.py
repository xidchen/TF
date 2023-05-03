import os
import keras_nlp
import tensorflow as tf


# Settings & hyperparameters

# Data
BATCH_SIZE = 64
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 450

# Model
EMBED_DIM = 256
FEED_FORWARD_DIM = 1024
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000

# Training
EPOCHS = 6

# Inference
NUM_TOKENS_TO_GENERATE = 80


# Load the data

tf.keras.utils.get_file(
    origin='https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip',
    extract=True
)
data_dir = os.path.expanduser('~/.keras/datasets/simplebooks/')
raw_train_ds = (
    tf.data.TextLineDataset(data_dir + 'simplebooks-92-raw/train.txt')
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)
raw_val_ds = (
    tf.data.TextLineDataset(data_dir + 'simplebooks-92-raw/valid.txt')
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
)


# Train the tokenizer

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    data=raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=['[PAD]', '[UNK]', '[BOS]'],
)


# Load tokenizer

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)


# Tokenize data

start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id('[BOS]'),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


train_ds = (
    raw_train_ds.map(map_func=preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
val_ds = (
    raw_val_ds.map(map_func=preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# Build the model

model_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)
layer_outputs = embedding_layer(model_inputs)

for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=FEED_FORWARD_DIM,
        num_heads=NUM_HEADS,
    )
    layer_outputs = decoder_layer(layer_outputs)

model_outputs = tf.keras.layers.Dense(units=VOCAB_SIZE)(layer_outputs)

model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer='adam', loss=loss_fn, metrics=[perplexity])
model.summary()


# Training

model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)


# Inference

prompt_tokens = tf.convert_to_tensor(
    value=[tokenizer.token_to_id(token='[BOS]')]
)


def token_logits_fn(inputs):
    cur_len = inputs.shape[1]
    output = model(inputs)
    return output[:, cur_len - 1, :]


# Greedy search
output_tokens = keras_nlp.utils.greedy_search(
    token_probability_fn=token_logits_fn,
    prompt=prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
)
generated_text = tokenizer.detokenize(output_tokens)
print(f'Greedy search generated text: \n{generated_text}\n')


# Beam search
output_tokens = keras_nlp.utils.beam_search(
    token_probability_fn=token_logits_fn,
    prompt=prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    num_beams=10,
    from_logits=True,
)
generated_text = tokenizer.detokenize(inputs=output_tokens)
print(f'Beam search generated text: \n{generated_text}\n')


# Random search
output_tokens = keras_nlp.utils.random_search(
    token_probability_fn=token_logits_fn,
    prompt=prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    from_logits=True,
)
generated_text = tokenizer.detokenize(output_tokens)
print(f'Random search generated text: \n{generated_text}\n')


# Top-K search
output_tokens = keras_nlp.utils.top_k_search(
    token_probability_fn=token_logits_fn,
    prompt=prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    k=10,
    from_logits=True,
)
generated_text = tokenizer.detokenize(output_tokens)
print(f'Top-K search generated text: \n{generated_text}\n')


# Top-P search
output_tokens = keras_nlp.utils.top_p_search(
    token_probability_fn=token_logits_fn,
    prompt=prompt_tokens,
    max_length=NUM_TOKENS_TO_GENERATE,
    p=0.5,
    from_logits=True,
)
generated_text = tokenizer.detokenize(output_tokens)
print(f'Top-P search generated text: \n{generated_text}\n')


# Using callbacks for text generation

class TopKTextGenerator(tf.keras.callbacks.Callback):

    def __init__(self, k):
        super().__init__()
        self.k = k

    def on_epoch_end(self, epoch, logs=None):
        out_tokens = keras_nlp.utils.top_k_search(
            token_probability_fn=token_logits_fn,
            prompt=prompt_tokens,
            max_length=NUM_TOKENS_TO_GENERATE,
            k=self.k,
            from_logits=True,
        )
        txt = tokenizer.detokenize(out_tokens)
        print(f'\n\nTop-K search generated text: \n{txt}\n')


class TopPTextGenerator(tf.keras.callbacks.Callback):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def on_epoch_end(self, epoch, logs=None):
        out_tokens = keras_nlp.utils.top_p_search(
            token_probability_fn=token_logits_fn,
            prompt=prompt_tokens,
            max_length=NUM_TOKENS_TO_GENERATE,
            p=self.p,
            from_logits=True,
        )
        txt = tokenizer.detokenize(out_tokens)
        print(f'\n\nTop-P search generated text: \n{txt}\n')


text_generation_callback = TopKTextGenerator(k=10)
model.fit(train_ds.take(1), epochs=2, callbacks=[text_generation_callback])

text_generation_callback = TopPTextGenerator(p=0.5)
model.fit(train_ds.take(1), epochs=2, callbacks=[text_generation_callback])
