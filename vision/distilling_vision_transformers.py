import tensorflow as tf
import tensorflow_datasets as tfds

tf.keras.utils.set_random_seed(seed=42)


# Hyperparameters

# Model
MODEL_TYPE = 'deit_distilled_tiny_patch16_224'
RESOLUTION = 224
PATCH_SIZE = 16
NUM_PATCHES = (RESOLUTION // PATCH_SIZE) ** 2
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 192
NUM_HEADS = 3
NUM_LAYERS = 12
MLP_UNITS = [
    PROJECTION_DIM * 4,
    PROJECTION_DIM,
]
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.1

# Training
NUM_EPOCHS = 40
BASE_LR = 0.0005
WEIGHT_DECAY = 0.0001

# Data
BATCH_SIZE = 256
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 5


# Load the tf_flowers dataset and prepare preprocessing utilities

def preprocess_dataset(is_training=True):
    def fn(image, label):
        if is_training:
            # Resize to a bigger spatial resolution and take the random crops
            image = tf.image.resize(image, (RESOLUTION + 20, RESOLUTION + 20))
            image = tf.image.random_crop(image, (RESOLUTION, RESOLUTION, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
        label = tf.one_hot(indices=label, depth=NUM_CLASSES)
        return image, label
    return fn


def prepare_dataset(dataset: tf.data.Dataset, is_training=True):
    if is_training:
        dataset = dataset.shuffle(buffer_size=BATCH_SIZE * 10)
    dataset = dataset.map(
        map_func=preprocess_dataset(is_training), num_parallel_calls=AUTO,
    )
    return dataset.batch(batch_size=BATCH_SIZE).prefetch(buffer_size=AUTO)


train_dataset, valid_dataset = tfds.load(
    name='tf_flowers', split=['train[:90%]', 'train[90%:]'], as_supervised=True,
)
print(f'Number of training examples: {train_dataset.cardinality()}')
print(f'Number of validation examples: {valid_dataset.cardinality()}')
train_dataset = prepare_dataset(train_dataset, is_training=True)
valid_dataset = prepare_dataset(valid_dataset, is_training=False)


# Implementing the DeiT variants of ViT

# Since DeiT is an extension of ViT it'd make sense to first implement ViT
# and then extend it to support DeiT's components.
# First, we'll implement a layer for Stochastic Depth (Huang et al.)
# which is used in DeiT for regularization.
class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=True, **kwargs):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(inputs)[0],) + (1,) * (len(tf.shape(inputs)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return inputs / keep_prob * random_tensor
        return inputs


# We'll implement the MLP and Transformer blocks.
def mlp(x, dropout_rate, hidden_units):
    for idx, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units=units,
            activation=tf.nn.gelu if idx == 0 else None,
        )(x)
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def transformer(drop_prob, name):
    """Transformer block with pre-norm"""
    num_patches = (
        NUM_PATCHES + 2 if 'distilled' in MODEL_TYPE else NUM_PATCHES + 1
    )
    encoded_patches = tf.keras.Input(shape=(num_patches, PROJECTION_DIM))
    x1 = tf.keras.layers.LayerNormalization(
        epsilon=LAYER_NORM_EPS
    )(encoded_patches)
    att_output = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=DROPOUT_RATE,
    )(x1, x1)
    att_output = StochasticDepth(
        drop_prob=drop_prob
    )(att_output) if drop_prob else att_output
    x2 = tf.keras.layers.Add()([att_output, encoded_patches])
    x3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)
    x4 = mlp(x3, dropout_rate=DROPOUT_RATE, hidden_units=MLP_UNITS)
    x4 = StochasticDepth(drop_prob=drop_prob)(x4) if drop_prob else x4
    block_output = tf.keras.layers.Add()([x2, x4])
    return tf.keras.Model(encoded_patches, block_output, name=name)


# We'll now implement a ViTClassifier class building on top of the components
# we just developed. Here we'll be following the original pooling strategy
# used in the ViT paper -- use a class token and use the feature representations
# corresponding to it for classification.
class ViTClassifier(tf.keras.Model):
    """Vision Transformer base class"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Patchify + linear projection + reshaping
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=PROJECTION_DIM,
                kernel_size=(PATCH_SIZE, PATCH_SIZE),
                strides=(PATCH_SIZE, PATCH_SIZE),
                padding='VALID',
                name='conv_projection',
            ),
            tf.keras.layers.Reshape(
                target_shape=(NUM_PATCHES, PROJECTION_DIM),
                name='flatten_projection',
            ),
        ], name='projection')
        # Positional embedding
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros(shape=(1, NUM_PATCHES + 1, PROJECTION_DIM)),
            name='pos_embedding'
        )
        # Transformer block
        dpr = [x for x in tf.linspace(
            start=0.0, stop=DROP_PATH_RATE, num=NUM_LAYERS,
        )]
        self.transformer_blocks = [
            transformer(drop_prob=dpr[i], name=f'transformer_block_{i}')
            for i in range(NUM_LAYERS)
        ]
        # CLS token
        initial_value = tf.zeros(shape=(1, 1, PROJECTION_DIM))
        self.cls_token = tf.Variable(
            initial_value=initial_value, trainable=True, name='cls'
        )
        # Other layers
        self.dropout = tf.keras.layers.Dropout(rate=DROPOUT_RATE)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPS
        )
        self.head = tf.keras.layers.Dense(
            units=NUM_CLASSES,
            name='classification_head',
        )

    def call(self, inputs, training=None, mask=None):
        # Create patches and project the patches
        projected_patches = self.projection(inputs)
        # Append class token if needed
        cls_token = tf.tile(
            input=self.cls_token,
            multiples=(tf.shape(input=inputs)[0], 1, 1)
        )
        cls_token = tf.cast(cls_token, dtype=projected_patches.dtype)
        projected_patches = tf.concat(
            values=[cls_token, projected_patches], axis=1
        )
        # Add positional embeddings to the projected patches
        encoded_patches = (
            self.positional_embedding + projected_patches
        )
        encoded_patches = self.dropout(encoded_patches)
        # Iterate over the number of layers and stack up blocks of Transformer
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        # Final layer normalization
        representation = self.layer_norm(encoded_patches)
        # Pool representation
        encoded_patches = representation[:, 0]
        # Classification head
        output = self.head(encoded_patches)
        return output


# This class can be used standalone as ViT and is end-to-end trainable.
# Just remove the distilled phrase in MODEL_TYPE, and it should work
# with vit_tiny = ViTClassifier(). Let's now extend it to DeiT.

# Apart from the class token, DeiT has another token for distillation.
# During distillation, the logits corresponding to the class token are compared
# to the true labels, and the logits corresponding to the distillation token
# are compared to the teacher's predictions.
class ViTDistilled(ViTClassifier):

    def __init__(self, regular_training=False, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = 2
        self.regular_training = regular_training
        # CLS and distillation tokens, positional embedding
        init_value = tf.zeros(shape=(1, 1, PROJECTION_DIM))
        self.dist_token = tf.Variable(
            initial_value=init_value, name='dist_token',
        )
        self.positional_embedding = tf.Variable(
            initial_value=tf.zeros(
                shape=(1, NUM_PATCHES + self.num_tokens, PROJECTION_DIM),
            ),
            name='pos_embedding',
        )
        # Head layers
        self.head_dist = tf.keras.layers.Dense(
            units=NUM_CLASSES, name='distillation_head',
        )

    def call(self, inputs, training=None, mask=None):
        # Create patches and project the patches
        projected_patches = self.projection(inputs)
        # Append the tokens
        cls_token = tf.tile(self.cls_token, (tf.shape(inputs)[0], 1, 1))
        dist_token = tf.tile(self.dist_token, (tf.shape(inputs)[0], 1, 1))
        cls_token = tf.cast(cls_token, dtype=projected_patches.dtype)
        dist_token = tf.cast(dist_token, dtype=projected_patches.dtype)
        projected_patches = tf.concat(
            values=[cls_token, dist_token, projected_patches], axis=1
        )
        # Add positional embeddings to the projected patches
        encoded_patches = (self.positional_embedding + projected_patches)
        encoded_patches = self.dropout(encoded_patches)
        # Iterate over the number of layers and stack up blocks of Transformer
        for transformer_module in self.transformer_blocks:
            encoded_patches = transformer_module(encoded_patches)
        # Final layer normalization
        representation = self.layer_norm(encoded_patches)
        # Classification heads
        x, x_dist = (
            self.head(representation[:, 0]),
            self.head_dist(representation[:, 1]),
        )
        if not training or self.regular_training:
            return (x + x_dist) / 2
        elif training:
            return x, x_dist


# Implementing the trainer

class DeiT(tf.keras.Model):

    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.student_loss_tracker = tf.keras.metrics.Mean(name='student_loss')
        self.dist_loss_tracker = tf.keras.metrics.Mean(name='distillation_loss')
        self.student_loss_fn = None
        self.distillation_loss_fn = None

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.student_loss_tracker)
        metrics.append(self.dist_loss_tracker)
        return metrics

    def compile(
        self,
        optimizer=None,
        metrics=None,
        student_loss_fn=None,
        distillation_loss_fn=None,
        **kwargs,
    ):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        x, y = data
        # Forward pass of teacher
        teacher_predictions = tf.nn.softmax(
            logits=self.teacher(x, training=False), axis=-1
        )
        teacher_predictions = tf.argmax(input=teacher_predictions, axis=-1)
        with tf.GradientTape() as tape:
            # Forward pass of student
            cls_predictions, dist_predictions = self.student(
                x / 255.0, training=True
            )
            # Compute losses
            student_loss = self.student_loss_fn(y, cls_predictions)
            distillation_loss = self.distillation_loss_fn(
                teacher_predictions, dist_predictions
            )
            loss = (student_loss + distillation_loss) / 2
        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(target=loss, sources=trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`
        student_predictions = (cls_predictions + dist_predictions) / 2
        self.compiled_metrics.update_state(y, student_predictions)
        self.dist_loss_tracker.update_state(values=distillation_loss)
        self.student_loss_tracker.update_state(values=student_loss)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        return results

    def test_step(self, data):
        x, y = data
        # Compute predictions
        y_prediction = self.student(x / 255.0, training=False)
        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        # Update the metrics
        self.compiled_metrics.update_state(y, y_prediction)
        self.student_loss_tracker.update_state(values=student_loss)
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs, training=None, mask=None):
        return self.student(inputs / 255.0, training=False)


# Load the teacher model

# !wget https://github.com/sayakpaul/deit-tf/releases/download/v0.1.0/bit_teacher_flowers.zip
# !unzip bit_teacher_flowers.zip

bit_teacher_flowers = tf.keras.models.load_model('bit_teacher_flowers')


# Training through distillation

deit_tiny = ViTDistilled()
deit_distiller = DeiT(student=deit_tiny, teacher=bit_teacher_flowers)

lr_scaled = BASE_LR / 512 * BATCH_SIZE
deit_distiller.compile(
    optimizer=tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_scaled, weight_decay=WEIGHT_DECAY
    ),
    metrics=['accuracy'],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1
    ),
    distillation_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),
)
history = deit_distiller.fit(
    train_dataset, epochs=NUM_EPOCHS, validation_data=valid_dataset,
)
