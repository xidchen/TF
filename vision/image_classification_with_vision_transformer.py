import tensorflow as tf


# Prepare the data

NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
print(f'x_train shape: {x_train.shape} - y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape} - y_test shape: {y_test.shape}')


# Configure the hyperparameters

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 100
IMAGE_SIZE = 72  # we'll resize input images to this size
PATCH_SIZE = 6  # size of the patches to be extracted
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]  # size of the transformer layers
TRANSFORMER_LAYERS = 8
MLP_HEADS_UNITS = [2048, 1024]  # size of the final dense layers


# Use data augmentation

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Normalization(),
    tf.keras.layers.Resizing(height=IMAGE_SIZE, width=IMAGE_SIZE),
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomRotation(factor=0.02),
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
], name='data_augmentation')
# compute the mean and variance of the training data for normalization
data_augmentation.layers[0].adapt(x_train)


# Implement multilayer perceptron (MLP)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


# Implement patch creation as a layer

class Patches(tf.keras.layers.Layer):

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images, *args, **kwargs):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
            name='patches'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, shape=[batch_size, -1, patch_dims])
        return patches


# Implement the patch encoding layer

# The PatchEncoder layer will linearly transform a patch by projecting it
# into a vector of size projection_dim.
# In addition, it adds a learnable position embedding to the projected vector.

class PatchEncoder(tf.keras.layers.Layer):

    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch, *args, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Build the ViT model

# The ViT model consists of multiple Transformer blocks,
# which use the layers.MultiHeadAttention layer as a self-attention mechanism
# applied to the sequence of patches.
# The Transformer blocks produce a [batch_size, num_patches, projection_dim]
# tensor, which is processed via a classifier head with softmax
# to produce the final class probabilities output.

# Unlike the technique described in the paper,
# which prepends a learnable embedding to the sequence of encoded patches
# to serve as the image representation,
# all the outputs of the final Transformer block are reshaped
# with layers.Flatten() and used as the image representation input
# to the classifier head. Note that the layers.GlobalAveragePooling1D layer
# could also be used instead to aggregate the outputs of the Transformer block,
# especially when the number of patches and the projection dimensions are large.

def create_vit_classifier():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    augmented = data_augmentation(inputs)
    patches = Patches(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)
    for _ in range(TRANSFORMER_LAYERS):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3, x2])
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)  # why 0.5?
    features = mlp(representation, hidden_units=MLP_HEADS_UNITS, dropout_rate=0.5)
    logits = tf.keras.layers.Dense(NUM_CLASSES)(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


# Compile, train, and evaluate the mode

def run_experiment(model: tf.keras.Model):
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
        ]
    )
    checkpoint_filepath = '/tmp/checkpoint'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
    )
    model.summary()
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=2,
        callbacks=[checkpoint_callback],
        validation_split=0.1,
    )
    model.load_weights(filepath=checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x=x_test, y=y_test)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')
    print(f'Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%')
    return history


vit_classifier = create_vit_classifier()
experiment_history = run_experiment(vit_classifier)


# After 100 epochs, the ViT model achieves around 55% accuracy and
# 82% top-5 accuracy on the test data.
# These are not competitive results on the CIFAR-100 dataset,
# as a ResNet50V2 trained from scratch on the same data can achieve 67% accuracy.

# Note that the state-of-the-art results reported in the paper are achieved
# by pre-training the ViT model using the JFT-300M dataset,
# then fine-tuning it on the target dataset.
# To improve the model quality without pre-training,
# you can try to train the model for more epochs,
# use a larger number of Transformer layers, resize the input images,
# change the patch size, or increase the projection dimensions.
# Besides, as mentioned in the paper, the quality of the model is affected
# not only by architecture choices, but also by parameters
# such as the learning rate schedule, optimizer, weight decay, etc.
# In practice, it's recommended to fine-tune a ViT model that was pre-trained
# using a large, high-resolution dataset.
