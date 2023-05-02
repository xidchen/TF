import abc
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# Hyperparameters

# data
DATASET_NAME = 'oxford_flowers102'
DATASET_REPETITIONS = 5
NUM_EPOCHS = 50
IMAGE_SIZE = 64
KID_IMAGE_SIZE = 75
KID_DIFFUSION_STEPS = 5
PLOT_DIFFUSION_STEPS = 20

# sampling
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

# architecture
EMBEDDING_DIMS = 32
EMBEDDING_MAX_FREQUENCY = 1000.0
WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2

# optimization
BATCH_SIZE = 64
EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# Data pipeline

def preprocess_image(data):
    # center crop image
    height = tf.shape(data['image'])[0]
    width = tf.shape(data['image'])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data['image'],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )
    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tfds
        .load(DATASET_NAME, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(DATASET_REPETITIONS)
        .shuffle(10 * BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_dataset = prepare_dataset('train[:80%]+validation[:80%]+test[:80%]')
val_dataset = prepare_dataset('train[80%:]+validation[80%:]+test[80%:]')


# Kernel inception distance

class KID(tf.keras.metrics.Metric, abc.ABC):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = tf.keras.metrics.Mean(name='kid_tracker')
        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
                tf.keras.layers.Rescaling(255.0),
                tf.keras.layers.Resizing(height=KID_IMAGE_SIZE, width=KID_IMAGE_SIZE),
                tf.keras.layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    weights='imagenet',
                    input_shape=(KID_IMAGE_SIZE, KID_IMAGE_SIZE, 3),
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
            ],
            name='inception_encoder',
        )

    @staticmethod
    def polynomial_kernel(features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)
        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)
        # estimate the squared maximum mean discrepancy using the average kernal values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(
            kernel_real * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


# Network architecture

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(EMBEDDING_MAX_FREQUENCY),
            EMBEDDING_DIMS // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def residual_block(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding='same',
                                   activation=tf.keras.activations.swish)(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.Add()(x, residual)
        return x
    return apply


def down_block(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x
    return apply


def up_block(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = residual_block(width)(x)
        return x
    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = tf.keras.Input(shape=(image_size, image_size, 3))
    noise_variances = tf.keras.Input(shape=(1, 1, 1))
    e = tf.keras.layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = tf.keras.layers.UpSampling2D(size=image_size, interpolation='nearest')(e)
    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = tf.keras.layers.Concatenate()([x, e])
    skips = []
    for width in widths[:-1]:
        x = down_block(width, block_depth)([x, skips])
    for _ in range(block_depth):
        x = residual_block(widths[-1])(x)
    for width in reversed(widths[:-1]):
        x = up_block(width, block_depth)([x, skips])
    x = tf.keras.layers.Conv2D(3, kernel_size=1, kernel_initializer='zeros')(x)
    return tf.keras.Model([noisy_images, noise_variances], x, name='residual_unet')


# Diffusion Model
class DiffusionModel(tf.keras.Model):

    def __init__(self, image_size, widths, block_depth):
        super().__init__()
        self.normalizer = tf.keras.layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = tf.keras.models.clone_model(self.network)
        self.noise_loss_tracker = None
        self.image_loss_tracker = None
        self.kid = None

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = tf.keras.metrics.Mean(name='n_loss')
        self.image_loss_tracker = tf.keras.metrics.Mean(name='i_loss')
        self.kid = KID(name='kid')

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=1.0)

    @staticmethod
    def diffusion_schedule(diffusion_times):
        # diffusion_times -> angles
        start_angle = tf.acos(MAX_SIGNAL_RATE)
        end_angle = tf.acos(MIN_SIGNAL_RATE)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        network = self.network if training else self.ema_network
        # predict noise component and calculate the image componenet using it
        pred_noises = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        # important line
        # at the first sampling step, the 'noisy image' is pure noise
        # but its signal rate is assumed to be nonzero (min_singal_rate)
        next_noisy_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            # network used in eval mode
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            # this new noise image will be used in the next step
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )
            # used for training
            noise_loss = self.loss(noises, pred_noises)
            # only used for metric
            image_loss = self.loss(images, pred_images)
        gradients = tape.gradient(target=noise_loss, sources=self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)
        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises
        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=BATCH_SIZE, diffusion_steps=KID_DIFFUSION_STEPS
        )
        self.kid.update_state(
            real_images=images, generated_images=generated_images
        )
        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, num_rows, num_cols):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=PLOT_DIFFUSION_STEPS,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()


# Training

# create and compile the model
model = DiffusionModel(image_size=IMAGE_SIZE, widths=WIDTHS, block_depth=BLOCK_DEPTH)
model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
), loss=tf.keras.losses.MAE)

# save the best model based on the validation KID metric
checkpoint_path = 'checkpoints/diffusion_model'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_kid',
    save_best_only=True,
    save_weights_only=True,
    mode='min',
)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(data=train_dataset)

# run training and plot generated images periodically
model.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=model.plot_images(num_rows=3, num_cols=6)
        ),
        checkpoint_callback,
    ],
    validation_data=val_dataset,
)


# Inference

# load the best model and generate images
model.load_weights(checkpoint_path)
model.plot_images(num_rows=3, num_cols=6)
