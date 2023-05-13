import keras_nlp
import os
import random
import tensorflow as tf

tf.keras.utils.set_random_seed(42)


# Hyperparamters

BATCH_SIZE = 64
EPOCHS = 1  # should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 40  # maybe too short
SOURCE_VOCAB_SIZE = 15000
TARGET_VOCAB_SIZE = 15000
RESERVED_TOKENS = ['[PAD]', '[UNK]', '[START]', '[END]']

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


# Downloading the data

ROOT_DIR = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
zip_data_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
    extract=True
)
os.remove(zip_data_path)
DATA_PATH = os.path.join(ROOT_DIR, 'fra.txt')
print(f'Dataset is downloaded at {DATA_PATH}')


# Parsing the data
with open(DATA_PATH) as f:
    data_lines = f.read().split('\n')[:-1]
data_pairs = []
for line in data_lines:
    source_lang, target_lang = line.split('\t')
    source_lang = source_lang.lower()
    target_lang = target_lang.lower()
    data_pairs.append((source_lang, target_lang))

random.shuffle(data_pairs)
num_val_samples = int(0.15 * len(data_pairs))
num_train_samples = len(data_pairs) - 2 * num_val_samples
train_pairs = data_pairs[:num_train_samples]
val_pairs = data_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = data_pairs[num_train_samples + num_val_samples:]

print(f'{len(data_pairs)} total pairs')
print(f'{len(train_pairs)} training pairs')
print(f'{len(val_pairs)} validation pairs')
print(f'{len(test_pairs)} test pairs')


# Tokenizing the data

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(tensors=text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        data=word_piece_ds.batch(batch_size=1000).prefetch(buffer_size=2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab


source_lang_samples = [data_pair[0] for data_pair in data_pairs]
source_lang_vocab = train_word_piece(
    text_samples=source_lang_samples,
    vocab_size=SOURCE_VOCAB_SIZE,
    reserved_tokens=RESERVED_TOKENS
)
target_lang_samples = [data_pair[1] for data_pair in data_pairs]
target_lang_vocab = train_word_piece(
    text_samples=target_lang_samples,
    vocab_size=TARGET_VOCAB_SIZE,
    reserved_tokens=RESERVED_TOKENS
)

source_lang_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=source_lang_vocab, lowercase=False
)
target_lang_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=target_lang_vocab, lowercase=False
)

for i in range(10):
    source_input_example = data_pairs[i][0]
    source_tokens_example = source_lang_tokenizer.tokenize(source_input_example)
    print(f'Source sentence: {source_input_example}')
    source_tokens = [
        source_lang_tokenizer.detokenize([t]).numpy().decode()
        for t in source_tokens_example.numpy()
    ]
    print(f'Tokens: {source_tokens}')
    target_input_example = data_pairs[i][1]
    target_tokens_example = target_lang_tokenizer.tokenize(target_input_example)
    print(f'Target sentence: {target_input_example}')
    target_tokens = [
        target_lang_tokenizer.detokenize([t]).numpy().decode()
        for t in target_tokens_example.numpy()
    ]
    print(f'Tokens: {target_tokens}')
    print()


# Format datasets

def preprocess_batch(source_ds, target_ds):
    source_ds = source_lang_tokenizer(source_ds)
    target_ds = target_lang_tokenizer(target_ds)
    source_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=source_lang_tokenizer.token_to_id('[PAD]'),
    )  # why no need '[START]' and '[END]'?
    source_ds = source_start_end_packer(source_ds)
    target_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=target_lang_tokenizer.token_to_id('[START]'),
        end_value=target_lang_tokenizer.token_to_id('[END]'),
        pad_value=target_lang_tokenizer.token_to_id('[PAD]'),
    )
    target_ds = target_start_end_packer(target_ds)
    return (
        {
            'encoder_inputs': source_ds,
            'decoder_inputs': target_ds[:, :-1],
        },
        target_ds[:, 1:],
    )


def make_dataset(pairs):
    source_lang_texts, target_lang_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices(
        tensors=(list(source_lang_texts), list(target_lang_texts))
    )
    dataset = dataset.batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)


# Building the model

encoder_inputs = tf.keras.Input(
    shape=(None,), dtype=tf.int64, name='encoder_inputs'
)
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=SOURCE_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    embeddings_initializer='glorot_uniform',  # different initializer?
    mask_zero=True,
    name='encoder_token_and_position_embedding',
)(encoder_inputs)
encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM,
    num_heads=NUM_HEADS,
    dropout=0,  # change dropout?
    activation='relu',  # change activation?
    layer_norm_epsilon=1e-05,  # change epsilon?
    kernel_initializer='glorot_uniform',  # change initializer?
    name='transformer_encoder',
)(inputs=x)
encoder = tf.keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)

decoder_inputs = tf.keras.Input(
    shape=(None,), dtype=tf.int64, name='decoder_inputs'
)
encoded_seq_inputs = tf.keras.Input(
    shape=(None, EMBED_DIM), name='decoder_state_inputs'
)
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=TARGET_VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    embeddings_initializer='glorot_uniform',
    mask_zero=True,
    name='decoder_token_and_position_embedding',
)(decoder_inputs)
x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=INTERMEDIATE_DIM,
    num_heads=NUM_HEADS,
    dropout=0,
    activation='relu',
    layer_norm_epsilon=1e-05,
    kernel_initializer='glorot_uniform',
    name='transformer_decoder',
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = tf.keras.layers.Dropout(rate=0.5)(x)
decoder_outputs = tf.keras.layers.Dense(
    units=TARGET_VOCAB_SIZE, activation=tf.nn.softmax
)(x)
decoder = tf.keras.Model(
    inputs=[decoder_inputs, encoded_seq_inputs], outputs=decoder_outputs
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = tf.keras.Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=decoder_outputs,
    name='transformer'
)


# Training our model

transformer.summary()
transformer.compile(
    optimizer='rmsprop',  # change optimizers?
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)


# Decoding test sentences (qualitative analysis)

def decode_sequences(input_sentences):
    batch_size = tf.shape(input_sentences)[0]
    encoder_input_tokens = source_lang_tokenizer(input_sentences).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(decoder_input_tokens):
        return transformer(
            [encoder_input_tokens, decoder_input_tokens]
        )[:, -1, :]

    # Set the prompt to the '[START]' token.
    prompt = tf.fill(
        dims=(batch_size, 1), value=target_lang_tokenizer.token_to_id('[START]')
    )

    generated_tokens = keras_nlp.utils.top_p_search(
        token_probability_fn,
        prompt,
        p=0.1,
        max_length=40,
        end_token_id=target_lang_tokenizer.token_to_id('[END]'),
    )
    generated_sentences = target_lang_tokenizer.detokenize(generated_tokens)
    return generated_sentences


test_source_texts = [pair[0] for pair in test_pairs]
for i in range(10):
    input_sentence = random.choice(test_source_texts)
    translated = decode_sequences(tf.constant([input_sentence]))
    translated = translated.numpy()[0].decode('utf-8')
    translated = (
        translated.replace('[PAD]', '')
        .replace('[START]', '')
        .replace('[END]', '')
        .strip()
    )
    print(f'Input: {input_sentence}')
    print(f'Translation: {translated}')
    print()


# Evaluating our model (quantitative analysis)

rouge_1 = keras_nlp.metrics.RougeN(order=1)
rouge_2 = keras_nlp.metrics.RougeN(order=2)

for test_pair in test_pairs[:30]:
    input_sentence = test_pair[0]
    reference_sentence = test_pair[1]

    translated_sentence = decode_sequences(tf.constant([input_sentence]))
    translated_sentence = translated_sentence.numpy()[0].decode('utf-8')
    translated_sentence = (
        translated_sentence.replace('[PAD]', '')
        .replace('[START]', '')
        .replace('[END]', '')
        .strip()
    )

    rouge_1(reference_sentence, translated_sentence)
    rouge_2(reference_sentence, translated_sentence)

print('ROUGE-1 Score: ', rouge_1.result())
print('ROUGE-2 Score: ', rouge_2.result())
