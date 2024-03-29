import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


stable_diffusion = keras_cv.models.StableDiffusion(jit_compile=True)


def plot_images(imgs):
    plt.figure(figsize=(20, 20))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(imgs[i])
        plt.axis("off")


# Assembling a text-image pair dataset


def assemble_image_dataset(urls):
    files = [tf.keras.utils.get_file(origin=url) for url in urls]
    resize = tf.keras.layers.Resizing(
        height=512, width=512, crop_to_aspect_ratio=True
    )
    imgs = [tf.keras.utils.load_img(img) for img in files]
    imgs = [tf.keras.utils.img_to_array(img) for img in imgs]
    imgs = np.array([resize(img) for img in imgs])
    imgs = imgs / 127.5 - 1
    img_dataset = tf.data.Dataset.from_tensor_slices(imgs)
    img_dataset = img_dataset.shuffle(50, reshuffle_each_iteration=True)
    img_dataset = img_dataset.map(
        keras_cv.layers.RandomCropAndResize(
            target_size=(512, 512),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(1.0, 1.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    img_dataset = img_dataset.map(
        keras_cv.layers.RandomFlip(),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return img_dataset


MAX_PROMPT_LENGTH = 77
placeholder_token = "<my-funny-cat-token>"


def pad_embedding(embedding):
    return embedding + (
        [stable_diffusion.tokenizer.end_of_text] *
        (MAX_PROMPT_LENGTH - len(embedding))
    )


stable_diffusion.tokenizer.add_tokens(placeholder_token)


def assemble_text_dataset(prompts):
    prompts = [prompt.format(placeholder_token) for prompt in prompts]
    embeddings = [stable_diffusion.tokenizer.encode(prompt) for prompt in prompts]
    embeddings = [np.array(pad_embedding(embedding)) for embedding in embeddings]
    text_dataset = tf.data.Dataset.from_tensor_slices(embeddings)
    text_dataset = text_dataset.shuffle(100, reshuffle_each_iteration=True)
    return text_dataset


def assemble_dataset(urls, prompts):
    img_dataset = assemble_image_dataset(urls)
    text_dataset = assemble_text_dataset(prompts)
    img_dataset = img_dataset.repeat()
    text_dataset = text_dataset.repeat(5)
    return tf.data.Dataset.zip((img_dataset, text_dataset))


# On the importance of prompt accuracy

single_ds = assemble_dataset(
    urls=[
        "https://i.imgur.com/VIedH1X.jpg",
        "https://i.imgur.com/eBw13hE.png",
        "https://i.imgur.com/oJ3rSg7.png",
        "https://i.imgur.com/5mCL6Df.jpg",
        "https://i.imgur.com/4Q6WWyI.jpg",
    ],
    prompts=[
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ],
)

group_ds = assemble_dataset(
    urls=[
        "https://i.imgur.com/yVmZ2Qa.jpg",
        "https://i.imgur.com/JbyFbZJ.jpg",
        "https://i.imgur.com/CCubd3q.jpg",
    ],
    prompts=[
        "a photo of a group of {}",
        "a rendering of a group of {}",
        "a cropped photo of the group of {}",
        "the photo of a group of {}",
        "a photo of a clean group of {}",
        "a photo of my group of {}",
        "a photo of a cool group of {}",
        "a close-up photo of a group of {}",
        "a bright photo of the group of {}",
        "a cropped photo of a group of {}",
        "a photo of the group of {}",
        "a good photo of the group of {}",
        "a photo of one group of {}",
        "a close-up photo of the group of {}",
        "a rendition of the group of {}",
        "a photo of the clean group of {}",
        "a rendition of a group of {}",
        "a photo of a nice group of {}",
        "a good photo of a group of {}",
        "a photo of the nice group of {}",
        "a photo of the small group of {}",
        "a photo of the weird group of {}",
        "a photo of the large group of {}",
        "a photo of a cool group of {}",
        "a photo of a small group of {}",
    ],
)

train_ds = single_ds.concatenate(group_ds)
train_ds = train_ds.batch(1).shuffle(
    train_ds.cardinality(), reshuffle_each_iteration=True
)


# Adding a new token to the text encoder

tokenized_initializer = stable_diffusion.tokenizer.encode("cat")[1]
new_weights = stable_diffusion.text_encoder.layers[2].token_embedding(
    tf.constant(tokenized_initializer)
)

# Get len of .vocab instead of tokenizer
new_vocab_size = len(stable_diffusion.tokenizer.vocab)

# The embedding layer is the 2nd layer in the text encoder
old_token_weights = stable_diffusion.text_encoder.layers[
    2
].token_embedding.get_weights()
old_position_weights = stable_diffusion.text_encoder.layers[
    2
].position_embedding.get_weights()

old_token_weights = old_token_weights[0]
new_weights = np.expand_dims(new_weights, axis=0)
new_weights = np.concatenate([old_token_weights, new_weights], axis=0)


# Have to set download_weights False,
# so we can initialize (otherwise tries to load weights)
new_encoder = keras_cv.models.stable_diffusion.TextEncoder(
    keras_cv.models.stable_diffusion.stable_diffusion.MAX_PROMPT_LENGTH,
    vocab_size=new_vocab_size,
    download_weights=False,
)
for index, layer in enumerate(stable_diffusion.text_encoder.layers):
    # Layer 2 is the embedding layer, so we omit it from our weight-copying
    if index == 2:
        continue
    new_encoder.layers[index].set_weights(layer.get_weights())


new_encoder.layers[2].token_embedding.set_weights([new_weights])
new_encoder.layers[2].position_embedding.set_weights(old_position_weights)

stable_diffusion._text_encoder = new_encoder.compile(jit_compile=True)


# Training

stable_diffusion.diffusion_model.trainable = False
stable_diffusion.decoder.trainable = False
stable_diffusion.text_encoder.trainable = True

stable_diffusion.text_encoder.layers[2].trainable = True


def traverse_layers(_layer):
    if hasattr(_layer, "layers"):
        for _layer in _layer.layers:
            yield _layer
    if hasattr(_layer, "token_embedding"):
        yield _layer.token_embedding
    if hasattr(_layer, "position_embedding"):
        yield _layer.position_embedding


for layer in traverse_layers(stable_diffusion.text_encoder):
    if (isinstance(layer, tf.keras.layers.Embedding)
            or "clip_embedding" in layer.name):
        layer.trainable = True
    else:
        layer.trainable = False

new_encoder.layers[2].position_embedding.trainable = False

all_models = [
    stable_diffusion.text_encoder,
    stable_diffusion.diffusion_model,
    stable_diffusion.decoder,
]
print([[w.shape for w in model.trainable_weights] for model in all_models])


# Training the new embedding

# Remove the top layer from the encoder, which cuts off the variance
# and only returns the mean
training_image_encoder = tf.keras.Model(
    stable_diffusion.image_encoder.input,
    stable_diffusion.image_encoder.layers[-2].output,
)


def sample_from_encoder_outputs(outputs):
    mean, logvar = tf.split(outputs, 2, axis=-1)
    logvar = tf.clip_by_value(logvar, -30.0, 20.0)
    std = tf.exp(0.5 * logvar)
    sample = tf.random.normal(tf.shape(mean))
    return mean + std * sample


def get_timestep_embedding(timestep, dim=320, max_period=10000.0):
    half = dim // 2
    freqs = tf.math.exp(
        -tf.math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
    )
    args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
    embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
    return embedding


def get_position_ids():
    return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)


class StableDiffusionFineTuner(tf.keras.Model):
    def __init__(self, _stable_diffusion, _noise_scheduler, **kwargs):
        super().__init__(**kwargs)
        self.stable_diffusion = _stable_diffusion
        self.noise_scheduler = _noise_scheduler

    def train_step(self, data):
        images, embeddings = data

        with tf.GradientTape() as tape:
            # Sample from the predicted distribution for the training image
            latents = sample_from_encoder_outputs(training_image_encoder(images))
            # The latents must be downsampled to match the scale of the latents
            # used in the training of StableDiffusion.  This number is truly
            # just a "magic" constant that they chose when training the model.
            latents = latents * 0.18215

            # Produce random noise in the same shape as the latent sample
            noise = tf.random.normal(tf.shape(latents))
            batch_dim = tf.shape(latents)[0]

            # Pick a random timestep for each sample in the batch
            timesteps = tf.random.uniform(
                (batch_dim,),
                minval=0,
                maxval=self.noise_scheduler.train_timesteps,
                dtype=tf.int64,
            )

            # Add noise to the latents based on the timestep for each sample
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps
            )

            # Encode the text in the training samples to use as hidden state
            # in the diffusion model
            encoder_hidden_state = self.stable_diffusion.text_encoder(
                [embeddings, get_position_ids()]
            )

            # Compute timestep embeddings for the randomly-selected timesteps
            # for each sample in the batch
            timestep_embeddings = tf.map_fn(
                fn=get_timestep_embedding,
                elems=timesteps,
                fn_output_signature=tf.float32,
            )

            # Call the diffusion model
            noise_pred = self.stable_diffusion.diffusion_model(
                [noisy_latents, timestep_embeddings, encoder_hidden_state]
            )

            # Compute the mean-squared error loss and reduce it.
            loss = self.compiled_loss(noise_pred, noise)
            loss = tf.reduce_mean(loss, axis=2)
            loss = tf.reduce_mean(loss, axis=1)
            loss = tf.reduce_mean(loss)

        # Load the trainable weights and compute the gradients for them
        trainable_weights = self.stable_diffusion.text_encoder.trainable_weights
        grads = tape.gradient(loss, trainable_weights)

        # Gradients are stored in indexed slices, so we have to find the index
        # of the slice(s) which contain the placeholder token.
        tf.reshape(tf.where(grads[0].indices == 49408), ())
        condition = grads[0].indices == 49408
        condition = tf.expand_dims(condition, axis=-1)

        # Override the gradients, zeroing out the gradients for all slices that
        # aren't for the placeholder token, effectively freezing the weights
        # for all other tokens.
        grads[0] = tf.IndexedSlices(
            values=tf.where(condition, grads[0].values, 0),
            indices=grads[0].indices,
            dense_shape=grads[0].dense_shape,
        )

        self.optimizer.apply_gradients(zip(grads, trainable_weights))
        return {"loss": loss}


generated = stable_diffusion.text_to_image(
    f"an oil painting of {placeholder_token}",
    seed=1337,
    batch_size=3,
)
plot_images(generated)


noise_scheduler = keras_cv.models.stable_diffusion.NoiseScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    train_timesteps=1000,
)
trainer = StableDiffusionFineTuner(
    stable_diffusion,
    noise_scheduler,
    name="trainer"
)
EPOCHS = 50
learning_rate = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=train_ds.cardinality() * EPOCHS,
)
optimizer = tf. keras.optimizers.Adam(
    weight_decay=0.004,
    learning_rate=learning_rate,
    epsilon=1e-8,
    global_clipnorm=10,
)

trainer.compile(
    optimizer=optimizer,
    # We are performing reduction manually in our train step,
    # so none is required here.
    loss=tf.keras.losses.MeanSquaredError(reduction="none"),
)


class GenerateImages(tf.keras.callbacks.Callback):
    def __init__(
        self, _stable_diffusion, prompt, steps=50, frequency=10, seed=None
    ):
        super().__init__()
        self.stable_diffusion = _stable_diffusion
        self.prompt = prompt
        self.seed = seed
        self.frequency = frequency
        self.steps = steps

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            images = self.stable_diffusion.text_to_image(
                self.prompt, batch_size=3, num_steps=self.steps, seed=self.seed
            )
            plot_images(images)


cbs = [
    GenerateImages(
        stable_diffusion,
        prompt=f"an oil painting of {placeholder_token}",
        seed=1337,
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"gandalf the gray as a {placeholder_token}",
        seed=1337,
    ),
    GenerateImages(
        stable_diffusion,
        prompt=f"two {placeholder_token} getting married, photorealistic, high quality",
        seed=1337,
    ),
]

trainer.fit(train_ds, epochs=EPOCHS, callbacks=cbs)


# Taking the Fine-Tuned Model for a Spin
generated = stable_diffusion.text_to_image(
    f"Gandalf as a {placeholder_token} fantasy art drawn by disney concept artists, "
    "golden colour, high quality, highly detailed, elegant, sharp focus, concept art, "
    "character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(generated)

generated = stable_diffusion.text_to_image(
    f"A masterpiece of a {placeholder_token} crying out to the heavens. "
    f"Behind the {placeholder_token}, an dark, evil shade looms over it - sucking the "
    "life right out of it.",
    batch_size=3,
)
plot_images(generated)

generated = stable_diffusion.text_to_image(
    f"An evil {placeholder_token}.", batch_size=3
)
plot_images(generated)

generated = stable_diffusion.text_to_image(
    f"A mysterious {placeholder_token} approaches the great pyramids of egypt.",
    batch_size=3,
)
plot_images(generated)

generated = stable_diffusion.text_to_image(
    f"A mysterious {placeholder_token} approaches the great pyramids of egypt, "
    "colored, high quality, highly detailed, sharp focus",
    batch_size=3,
)
plot_images(generated)
