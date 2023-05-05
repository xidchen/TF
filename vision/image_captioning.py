import abc
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 111
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Download the dataset

# !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
# !wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
# !unzip -qq Flickr8k_Dataset.zip
# !unzip -qq Flickr8k_text.zip
# !rm Flickr8k_Dataset.zip Flickr8k_text.zip

# Path to the images
IMAGES_PATH = "Flicker8k_Dataset"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE


# Preparing the dataset

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()
        for line in caption_data:
            line = line.rstrip('\n')
            img_name, caption = line.split('\t')
            img_name = img_name.split('#')[0]
            img_name = os.path.join(IMAGES_PATH, img_name.strip())
            tokens = caption.strip().split()
            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue
            if img_name.endswith('jpg') and img_name not in images_to_skip:
                caption = '<start> ' + caption.strip() + ' <end>'
                text_data.append(caption)
                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]
        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]
        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)
    train_size = int(len(caption_data) * train_size)
    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }
    return training_data, validation_data


caption_mapping_dict, text_data_list = load_captions_data('Flickr8k.token.txt')
train_data, valid_data = train_val_split(caption_mapping_dict)
print(f'Number of training samples: {len(train_data)}')
print(f'Number of validation samples: {len(valid_data)}')


# Vectorizing the text data

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(strip_chars), '')


strip_chars = string.punctuation.replace('<', '').replace('>', '')
vectorization = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=custom_standardization,
    output_sequence_length=SEQ_LENGTH,
)
vectorization.adapt(data=text_data_list)

image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomRotation(factor=0.2),
    tf.keras.layers.RandomContrast(factor=0.3),
])


# Building a tf.data.Dataset pipeline for training

def decode_and_resize(img_path):
    img = tf.io.read_file(filename=img_path)
    img = tf.image.decode_jpeg(contents=img, channels=3)
    img = tf.image.resize(images=img, size=IMAGE_SIZE)
    img = tf.image.convert_image_dtype(image=img, dtype=tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices(tensors=(images, captions))
    dataset = dataset.shuffle(buffer_size=BATCH_SIZE * 8)
    dataset = dataset.map(map_func=process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))


# Building the model

# Our image captioning architecture consists of three models:
# A CNN: used to extract the image features
# A TransformerEncoder: The extracted image features are then passed to
# a Transformer based encoder that generates a new representation of the inputs
# A TransformerDecoder: This model takes the encoder output and the text data
# (sequences) as inputs and tries to learn to generate the caption.

def get_cnn_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = tf.keras.layers.Reshape(
        target_shape=(-1, base_model_out.shape[-1])
    )(base_model_out)
    return tf.keras.Model(inputs=base_model.inputs, outputs=base_model_out)


class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.dense_1 = tf.keras.layers.Dense(embed_dim, activation=tf.nn.relu)

    def call(self, inputs, training=None, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)
        attention_output_1 = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=None,
            training=training,
        )
        return self.layernorm_2(inputs + attention_output_1)


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs, *args, **kwargs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, 0)


class TransformerDecoderBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = tf.keras.layers.Dense(ff_dim, activation=tf.nn.relu)
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.embedding = PositionalEmbedding(
            sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM
        )
        self.out = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.nn.softmax)
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
        self.supports_masking = True

    @staticmethod
    def get_causal_attention_mask(inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, shape=(1, input_shape[1], input_shape[1]))
        mult = tf.concat(values=[
            tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)
        ], axis=0)
        return tf.tile(input=mask, multiples=mult)

    def call(self, inputs, encoder_outputs=None, training=None, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        padding_mask, combined_mask = None, None
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(out_2 + ffn_out, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        return self.out(ffn_out)


class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn, encoder, decoder, num_captions_per_image=5, image_aug=None):
        super().__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.Mean(name='accuracy')

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    @staticmethod
    def calculate_accuracy(y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(
            y_true=batch_seq_true, y_pred=batch_seq_pred, mask=mask
        )
        acc = self.calculate_accuracy(
            y_true=batch_seq_true, y_pred=batch_seq_pred, mask=mask
        )
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        if self.image_aug:
            batch_img = self.image_aug(batch_img)
        img_embed = self.cnn(batch_img)
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self.compute_caption_loss_and_acc(
                    img_embed=img_embed, batch_seq=batch_seq[:, i, :], training=True
                )
                batch_loss += loss
                batch_acc += acc
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )
            grads = tape.gradient(target=loss, sources=train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars))
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(values=batch_loss)
        self.acc_tracker.update_state(values=batch_acc)
        return {'loss': self.loss_tracker.result(), 'acc': self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_embed = self.cnn(batch_img)
        for i in range(self.num_captions_per_image):
            loss, acc = self.compute_caption_loss_and_acc(
                img_embed=img_embed, batch_seq=batch_seq[:, i, :], training=False
            )
            batch_loss += loss
            batch_acc += acc
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(values=batch_loss)
        self.acc_tracker.update_state(values=batch_acc)
        return {'loss': self.loss_tracker.result(), 'acc': self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


cnn_model = get_cnn_model()
encoder_model = TransformerEncoderBlock(
    embed_dim=EMBED_DIM,
    ff_dim=FF_DIM,
    num_heads=1
)
decoder_model = TransformerDecoderBlock(
    embed_dim=EMBED_DIM,
    ff_dim=FF_DIM,
    num_heads=2
)
caption_model = ImageCaptioningModel(
    cnn=cnn_model,
    encoder=encoder_model,
    decoder=decoder_model,
    image_aug=image_augmentation
)


# Model training

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none'
)  # why reduction='none'?
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True
)


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, abc.ABC):

    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, dtype=tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, dtype=tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            pred=global_step < warmup_steps,
            true_fn=lambda: warmup_learning_rate,
            false_fn=lambda: self.post_warmup_learning_rate
        )


num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(
    post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps
)

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=cross_entropy,
)
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=valid_dataset,
)


# Check sample predictions

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    sample_image = np.random.choice(valid_images)
    sample_image = decode_and_resize(img_path=sample_image)
    img = sample_image.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    # Pass the image to the CNN
    img = tf.expand_dims(input=sample_image, axis=0)
    img = caption_model.cnn(img)
    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)
    # Generate the caption using the Transformer decoder
    decoded_caption = '<start> '
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == ' <end>':
            break
        decoded_caption += ' ' + sampled_token
    decoded_caption = decoded_caption.replace('<start> ', '')
    decoded_caption = decoded_caption.replace(' <end>', '').strip()
    print(f'Predicted Caption: {decoded_caption}')


generate_caption()
generate_caption()
generate_caption()
