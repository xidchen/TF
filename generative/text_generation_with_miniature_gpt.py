# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz

import numpy as np
import os
import string
import tensorflow as tf


# Implement a Transformer block as a layer

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's from the lower triangle, counting from the lower right corner."""
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j  # The original m = i >= j + n_dest - n_src ends up with an error
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], tf.int32)], 0
    )
    return tf.tile(mask, mult)


class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, 'relu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def __call__(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        inputs = self.layernorm1(inputs)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = inputs + attention_output
        ffn_output = self.ffn(self.layernorm2(out1))
        ffn_output = self.dropout2(ffn_output)
        return out1 + ffn_output


# Implement an embedding layer

class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(maxlen, embed_dim)

    def __call__(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# Implement the miniature GPT model

VOCAB_SIZE = 20000
MAX_LEN = 80
EMBED_DIM = 256
NUM_HEADS = 4
FEED_FORWARD_DIM = 1024


def create_model():
    inputs = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FORWARD_DIM)
    x = transformer_block(x)
    outputs = tf.keras.layers.Dense(VOCAB_SIZE)(x)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=[loss_fn, None])
    return model


# Prepare the data for word-level language modelling

BATCH_SIZE = 128

filenames = []
directories = [
    'aclImdb/train/pos',
    'aclImdb/train/neg',
    'aclImdb/test/pos',
    'aclImdb/test/neg',
]
for directory in directories:
    for f in os.listdir(directory):
        filenames.append(os.path.join(directory, f))
print(f'{len(filenames)} files')

text_ds = tf.data.TextLineDataset(tf.random.shuffle(filenames))
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(BATCH_SIZE)


def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, '<br />', '')
    return tf.strings.regex_replace(
        stripped_html, f'([{string.punctuation}])', r' \1')


vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE - 1,
    standardize=custom_standardization,
    output_mode='int',
    output_sequence_length=MAX_LEN + 1,
)
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()


def prepare_lm_inputs_labels(text):
    """Shift word sequences by 1 position so that the target for position (i) is
    word at position (i + 1). The model will use all words up till position (i)
    to predict the next word.
    """
    text = tf.expand_dims(text, axis=-1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels)
text_ds = text_ds.prefetch(tf.data.AUTOTUNE)


# Implement a Keras callback for generating text

class TextGenerator(tf.keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model.
    2. Predict probabilities for the next token.
    3. Sample the next token and add it to the next input.

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1):
        super().__init__()
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.top_k = top_k
        self.print_every = print_every

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.top_k, sorted=True)
        indices = np.asarray(indices).astype('int32')
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype('float32')
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = MAX_LEN - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:MAX_LEN]
                sample_index = MAX_LEN - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = ' '.join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f'generated text:\n{txt}\n')


word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index
start_txt_prompt = 'this movie is'
start_txt_tokens = [word_to_index.get(_, 1) for _ in start_txt_prompt.split()]
number_tokens_generated = 40
text_gen_callback = TextGenerator(number_tokens_generated, start_txt_tokens, vocab)


# Train the model
text_gen_model = create_model()
text_gen_model.fit(text_ds, epochs=25, verbose=2, callbacks=[text_gen_callback])
