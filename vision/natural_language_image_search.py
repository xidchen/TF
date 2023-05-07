import collections
import json
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text
import tqdm


# Prepare the data

# We will use the MS-COCO dataset to train our dual encoder model.
# MS-COCO contains over 82,000 images, each of which has at least
# 5 different caption annotations.
# The dataset is usually used for image captioning tasks, but we can repurpose
# the image-caption pairs to train our dual encoder model for image search.

# First, let's download the dataset, which consists of two compressed folders:
# one with images, and the other with associated image captions.
# Note that the compressed images folder is 13GB in size.

ROOT_DIR = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
ANNOTATIONS_DIR = os.path.join(ROOT_DIR, 'annotations')
IMAGES_DIR = os.path.join(ROOT_DIR, 'train2014')
TFRECORDS_DIR = os.path.join(ROOT_DIR, 'tfrecords')
ANNOTATION_FILE = os.path.join(ANNOTATIONS_DIR, 'captions_train2014.json')

if not os.path.exists(ANNOTATIONS_DIR):
    annotation_zip = tf.keras.utils.get_file(
        origin='https://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        extract=True,
    )
    os.remove(annotation_zip)
if not os.path.exists(IMAGES_DIR):
    image_zip = tf.keras.utils.get_file(
        origin='https://images.cocodataset.org/zips/train2014.zip',
        extract=True,
    )
    os.remove(image_zip)
print('Dataset is downloaded and extracted successfully.')
with open(ANNOTATION_FILE, 'r') as f:
    annotations = json.load(f)['annotations']

image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption_name = f'{element["caption"].lower().rstrip(".")}'
    image_path = IMAGES_DIR + '/COCO_train2014_' + f'{element["image_id"]:012d}.jpg'
    image_path_to_caption[image_path].append(caption_name)
image_paths = list(image_path_to_caption.keys())
print(f'Number of images: {len(image_paths)}')

# Process and save the data to TFRecord files
# We set train_size to 30,000 images, which is about 35% of the dataset.
# We use 2 captions for each image, thus producing 60,000 image-caption pairs.
# The size of the training set affects the quality of the produced encoders,
# but more examples would lead to longer training time.

TRAIN_SIZE = 30000
VALID_SIZE = 5000
CAPTIONS_PER_IMAGE = 2
IMAGES_PER_FILE = 2000
IMAGE_SIZE = (299, 299)

train_image_paths = image_paths[:TRAIN_SIZE]
num_train_files = math.ceil(TRAIN_SIZE / IMAGES_PER_FILE)
train_file_prefix = os.path.join(TFRECORDS_DIR, 'train')

valid_image_paths = image_paths[-VALID_SIZE:]
num_valid_files = math.ceil(VALID_SIZE / IMAGES_PER_FILE)
valid_file_prefix = os.path.join(TFRECORDS_DIR, 'valid')

tf.io.gfile.makedirs(TFRECORDS_DIR)


def bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(img_path, caption):
    feature = {
        'caption': bytes_features(caption.decode()),
        'raw_image': bytes_features(tf.io.read_file(img_path).numpy()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, img_paths):
    caption_list = []
    img_path_list = []
    for img_path in img_paths:
        captions = image_path_to_caption[img_path][:CAPTIONS_PER_IMAGE]
        caption_list.extend(captions)
        img_path_list.extend([img_path] * CAPTIONS_PER_IMAGE)
    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(img_path_list)):
            example = create_example(
                img_path=img_path_list[example_idx],
                caption=caption_list[example_idx]
            )
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(img_paths, num_files, files_prefix):
    example_counter = 0
    for file_idx in tqdm.tqdm(range(num_files)):
        file_name = files_prefix + f'-{file_idx:02d}.tfrecord'
        start_idx = IMAGES_PER_FILE * file_idx
        end_idx = start_idx + IMAGES_PER_FILE
        example_counter += write_tfrecords(file_name, img_paths[start_idx:end_idx])
    return example_counter


train_example_count = write_data(train_image_paths, num_train_files, train_file_prefix)
print(f'{train_example_count} training examples were written to tfrecord files.')
valid_example_count = write_data(valid_image_paths, num_valid_files, valid_file_prefix)
print(f'{valid_example_count} training examples were written to tfrecord files.')

# Create tf.data.Dataset for training and evaluation

feature_description = {
    'caption': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'raw_image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
}


def read_example(example):
    features = tf.io.parse_single_example(
        serialized=example, features=feature_description
    )
    raw_images = features.pop('raw_image')
    features['image'] = tf.image.resize(
        images=tf.image.decode_jpeg(contents=raw_images, channels=3),
        size=IMAGE_SIZE,
    )
    return features


def get_dataset(file_pattern, batch_size):
    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(map_func=read_example,
             num_parallel_calls=tf.data.AUTOTUNE,
             deterministic=False,)
        .shuffle(buffer_size=batch_size * 10)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .batch(batch_size=batch_size)
    )


# Implement the projection head

# The projection head is used to transform the image and the text embeddings
# to the same embedding space with the same dimensionality.

def project_embeddings(
        embeddings, num_proj_layers, proj_dims, dropout_rates
):
    proj_embeddings = tf.keras.layers.Dense(units=proj_dims)(embeddings)
    for _ in range(num_proj_layers):
        x = tf.nn.gelu(features=proj_embeddings)
        x = tf.keras.layers.Dense(units=proj_dims)(x)
        x = tf.keras.layers.Dropout(rate=dropout_rates)(x)
        x = tf.keras.layers.Add()([proj_embeddings, x])
        proj_embeddings = tf.keras.layers.LayerNormalization()(x)
    return proj_embeddings


# Implement the text encoder

def create_text_encoder(
        num_proj_layers, proj_dims, dropout_rates, trainable=False
):
    preprocess = hub.KerasLayer(
        handle='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        name='text_en_uncased_preprocessing',
    )
    bert = hub.KerasLayer(
        handle='https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/2',
        name='bert_en_uncased_L-6_H-512_A-8',
    )
    bert.trainable = trainable
    inputs = tf.keras.Input(shape=(), name='text_input', dtype=tf.string)
    bert_inputs = preprocess(inputs)
    embeddings = bert(bert_inputs)['pooled_output']
    outputs = project_embeddings(
        embeddings, num_proj_layers, proj_dims, dropout_rates
    )
    return tf.keras.Model(inputs, outputs, name='text_encoder')


# Implement the vision encoder

def create_vision_encoder(
        num_proj_layers, proj_dims, dropout_rates, trainable=False
):
    base_encoder = tf.keras.applications.Xception(
        include_top=False, weights='imagenet', pooling='avg',
    )
    for layer in base_encoder.layers:
        layer.trainable = trainable
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name='image_input')
    base_encoder_inputs = tf.keras.applications.xception.preprocess_input(inputs)
    embeddings = base_encoder(base_encoder_inputs)
    outputs = project_embeddings(
        embeddings, num_proj_layers, proj_dims, dropout_rates
    )
    return tf.keras.Model(inputs, outputs, name='vision_encoder')


# Implement the dual encoder

# To calculate the loss, we compute the pairwise dot-product similarity
# between each caption_i and images_j in the batch as the predictions.
# The target similarity between caption_i and image_j is computed
# as the average of the (dot-product similarity between caption_i and caption_j)
# and (the dot-product similarity between image_i and image_j).
# Then, we use crossentropy to compute the loss
# between the targets and the predictions.

class DualEncoder(tf.keras.Model):

    def __init__(self, text_encoder, vision_encoder, temperature, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.temperature = temperature
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, inputs, training=False, mask=None):
        # Place each encoder on a separate GPU (if available).
        # TF will fall back on available devices if there are fewer than 2 GPUs.
        with tf.device(device_name='/gpu:0'):
            text_embeddings = self.text_encoder(
                inputs['caption'], training=training
            )
        with tf.device(device_name='/gpu:1'):
            vision_embeddings = self.vision_encoder(
                inputs['image'], training=training
            )
        return text_embeddings, vision_embeddings

    def calculate_loss(self, text_embeddings, vision_embeddings):
        logits = tf.matmul(
            text_embeddings, vision_embeddings, transpose_b=True
        ) / self.temperature
        captions_similarity = tf.matmul(
            text_embeddings, text_embeddings, transpose_b=True
        )
        images_similarity = tf.matmul(
            vision_embeddings, vision_embeddings, transpose_b=True
        )
        targets = tf.keras.activations.softmax(
            (captions_similarity + images_similarity) / (2 * self.temperature)
        )
        captions_loss = tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        images_loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        return (captions_loss + images_loss) / 2

    def train_step(self, data):
        # Forward pass
        with tf.GradientTape() as tape:
            text_embeddings, vision_embeddings = self(data, training=True)
            loss = self.calculate_loss(text_embeddings, vision_embeddings)
        # Backward pass
        gradients = tape.gradient(target=loss, sources=self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(values=loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data):
        text_embeddings, vision_embeddings = self(data, training=False)
        loss = self.calculate_loss(text_embeddings, vision_embeddings)
        self.loss_tracker.update_state(values=loss)
        return {'loss': self.loss_tracker.result()}


# Train the dual encoder model

# In this experiment, we freeze the base encoders for text and images,
# and make only the projection head trainable.

NUM_EPOCHS = 5  # In practice, train for at least 30 epochs
BATCH_SIZE = 256

caption_encoder = create_text_encoder(
    num_proj_layers=1, proj_dims=256, dropout_rates=0.1
)
image_encoder = create_vision_encoder(
    num_proj_layers=1, proj_dims=256, dropout_rates=0.1
)
dual_encoder = DualEncoder(
    text_encoder=caption_encoder, vision_encoder=image_encoder, temperature=0.05
)
dual_encoder.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(
        learning_rate=0.001, weight_decay=0.001,
    )  # why not default weight_decay?
)

# Note that training the model with 60,000 image-caption pairs,
# with a batch size of 256, takes around 12 minutes per epoch
# using a V100 GPU accelerator.
# If 2 GPUs are available, each epoch takes around 8 minutes.

print(f'Number of GPUs: {len(tf.config.list_physical_devices("GPU"))}')
print(f'Number of examples (caption-image pairs): {train_example_count}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Steps per epoch: {math.ceil(train_example_count / BATCH_SIZE)}')
train_dataset = get_dataset(
    file_pattern=os.path.join(TFRECORDS_DIR, 'train-*.tfrecord'),
    batch_size=BATCH_SIZE
)
valid_dataset = get_dataset(
    file_pattern=os.path.join(TFRECORDS_DIR, 'valid-*.tfrecord'),
    batch_size=BATCH_SIZE
)
# Create a learning rate schedule callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3,
)
# Create an early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True,
)
print()
history = dual_encoder.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[reduce_lr, early_stopping],
    validation_data=valid_dataset,
)
print()
print('Training completed. Saving caption, image, and dual encoders...')
caption_encoder.save(filepath='caption_encoder', include_optimizer=False)
image_encoder.save(filepath='image_encoder', include_optimizer=False)
dual_encoder.save(filepath='dual_encoder', include_optimizer=False)
print('Models are saved.')
print()

# Polt the training loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()


# Search for images using natural language queries

# We can then retrieve images corresponding to natural language queries
# via the following steps:
#   1. Generate embeddings for the images by feeding them into the image_encoder.
#   2. Feed the natural language query to the caption_encoder
#      to generate a query embedding.
#   3. Compute the similarity between the query embedding and the image embeddings
#      in the index to retrieve the indices of the top matches.
#   4. Look up the paths of the top matching images to display them.
# Note that, after training the dual encoder,
# only the fine-tuned image_encoder and caption_encoder models will be used,
# while the dual_encoder model will be discarded.

# Generate embeddings for the images

# We load the images and feed them into the image_encoder
# to generate their embeddings. In large scale systems, this step is performed
# using a parallel data processing framework, such as Apache Spark or Apache Beam.
# Generating the image embeddings may take several minutes.

# print('Loading image and caption encoders...')
# image_encoder = tf.keras.models.load_model('image_encoder')
# caption_encoder = tf.keras.models.load_model('caption_encoder')
# print('Models are loaded.')

def read_image(img_path):
    img_array = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
    return tf.image.resize(images=img_array, size=IMAGE_SIZE)


print(f'Generating embeddings for {len(image_paths)} images...')
image_embeddings = image_encoder.predict(
    tf.data.Dataset.from_tensor_slices(image_paths).map(
        map_func=read_image, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(batch_size=BATCH_SIZE),
    verbose=1,
)
print(f'Image embeddings shape: {image_embeddings.shape}.')

# Retrieve relevant images

# We use exact matching by computing the dot product similarity
# between the input query embedding and the image embeddings,
# and retrieve the top k matches. However, approximate similarity matching,
# using frameworks like ScaNN, Annoy, or Faiss is preferred
# in real-time use cases to scale with a large number of images.


def find_matches(img_embeddings, queries, k=9, normalize=True):
    query_embedding = caption_encoder(tf.convert_to_tensor(queries))
    if normalize:
        img_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)
    dot_similarity = tf.matmul(query_embedding, img_embeddings, transpose_b=True)
    results = tf.math.top_k(input=dot_similarity, k=k).indices.numpy()
    return [[image_paths[idx] for idx in indices] for indices in results]


query = 'a family standing next to the ocean on a sandy beach with a surf board'
matches = find_matches(image_embeddings, [query])[0]

plt.figure(figsize=(20, 20))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(mpimg.imread(matches[i]))
    plt.axis('off')
plt.show()


# Evaluate the retrieval quality

# To evaluate the dual encoder model, we use the captions as queries.
# We use the out-of-training-sample images and captions
# to evaluate the retrieval quality, using top k accuracy.
# A true prediction is counted if, for a given caption,
# its associated image is retrieved within the top k matches.

def compute_top_k_accuracy(img_paths, k=100):
    hits = 0
    num_batches = math.ceil(len(img_paths) / BATCH_SIZE)
    for idx in tqdm.tqdm(range(num_batches)):
        start_idx = idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        current_img_paths = img_paths[start_idx:end_idx]
        queries = [
            image_path_to_caption[img_path][0] for img_path in current_img_paths
        ]
        result = find_matches(image_embeddings, queries, k)
        hits += sum([
            img_path in _matches
            for (img_path, _matches) in list(zip(current_img_paths, result))
        ])
    return hits / len(img_paths)


print()
print('Scoring training data...')
train_accuracy = compute_top_k_accuracy(img_paths=train_image_paths)
print(f'Train accuracy: {round(train_accuracy * 100, 3)}%')
print()
print('Scoring validation data...')
valid_accuracy = compute_top_k_accuracy(img_paths=valid_image_paths)
print(f'Valid accuracy: {round(valid_accuracy * 100, 3)}%')
print()


# You can obtain better results by increasing the size of the training sample,
# train for more epochs, explore other base encoders for images and text,
# set the base encoders to be trainable, and tune the hyperparameters,
# especially the temperature for the softmax in the loss computation.
