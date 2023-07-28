import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import textwrap


# Data loading

data_path = tf.keras.utils.get_file(
    origin="https://huggingface.co/datasets/sayakpaul/"
           "pokemon-blip-original-version/resolve/main/pokemon_dataset.tar.gz",
    untar=True,
)
data_frame = pd.read_csv(os.path.join(data_path, "data.csv"))
data_frame["image_path"] = data_frame["image_path"].apply(
    lambda x: os.path.join(data_path, x)
)
data_frame.head()
# Downloading data from ...
# 347504640/347504640 [==============================] - 2s 0us/step
#
# index	image_path	caption
# 0	/root/.keras/datasets/pokemon_dataset/image_0.png	a drawing of a green Pokémon with red eyes
# 1	/root/.keras/datasets/pokemon_dataset/image_1.png	a green and yellow toy with a red nose
# 2	/root/.keras/datasets/pokemon_dataset/image_2.png	a red and white ball with an angry look on its face
# 3	/root/.keras/datasets/pokemon_dataset/image_3.png	a cartoon ball with a smile on it's face
# 4	/root/.keras/datasets/pokemon_dataset/image_4.png	a bunch of balls with faces drawn on them


PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77

# Load the tokenizer.
tokenizer = keras_cv.models.stable_diffusion.SimpleTokenizer()
# Downloading data from https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true
# 1356917/1356917 [==============================] - 0s 0us/step


#  Method to tokenize and pad the tokens.
def process_text(text):
    tokens = tokenizer.encode(text)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
    return np.array(tokens)


# Collate the tokenized captions into an array.
tokenized_captions = np.empty((len(data_frame), MAX_PROMPT_LENGTH))

all_captions = data_frame["caption"].to_list()
for i, caption in enumerate(all_captions):
    tokenized_captions[i] = process_text(caption)


# Prepare a tf.data.Dataset

RESOLUTION = 256
AUTO = tf.data.AUTOTUNE
POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

augmenter = tf.keras.Sequential(
    layers=[
        tf.keras.layers.CenterCrop(RESOLUTION, RESOLUTION),
        keras_cv.layers.RandomFlip(),
        tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
    ]
)
text_encoder = keras_cv.models.stable_diffusion.TextEncoder(MAX_PROMPT_LENGTH)
# Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5
# 492466864/492466864 [==============================] - 4s 0us/step


def process_image(image_path, tokenized_text):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, 3)
    image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
    return image, tokenized_text


def apply_augmentation(image_batch, token_batch):
    return augmenter(image_batch), token_batch


def run_text_encoder(image_batch, token_batch):
    return (
        image_batch,
        token_batch,
        text_encoder([token_batch, POS_IDS], training=False),
    )


def prepare_dict(image_batch, token_batch, encoded_text_batch):
    return {
        "images": image_batch,
        "tokens": token_batch,
        "encoded_text": encoded_text_batch,
    }


def prepare_dataset(image_paths, tokenized_texts, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
    dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(batch_size)
    dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
    dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
    return dataset.prefetch(AUTO)


# Prepare the dataset.
training_dataset = prepare_dataset(
    np.array(data_frame["image_path"]), tokenized_captions, batch_size=4
)

# Take a sample batch and investigate.
sample_batch = next(iter(training_dataset))
for k in sample_batch:
    print(k, sample_batch[k].shape)
# images (4, 256, 256, 3)
# tokens (4, 77)
# encoded_text (4, 77, 768)


plt.figure(figsize=(20, 10))
for i in range(3):
    ax = plt.subplot(1, 4, i + 1)
    plt.imshow((sample_batch["images"][i] + 1) / 2)
    caption = tokenizer.decode(sample_batch["tokens"][i].numpy().squeeze())
    caption = caption.replace("<|startoftext|>", "")
    caption = caption.replace("<|endoftext|>", "")
    caption = "\n".join(textwrap.wrap(caption, 12))
    plt.title(caption, fontsize=15)
    plt.axis("off")


# A trainer class for the fine-tuning loop

class Trainer(tf.keras.Model):
    # Reference:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

    def __init__(
        self,
        diffusion_model,
        vae,
        noise_scheduler,
        use_mixed_precision=False,
        max_grad_norm=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        self.vae.trainable = False

    def train_step(self, inputs):
        images = inputs["images"]
        encoded_text = inputs["encoded_text"]
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as tape:
            # Project image into the latent space and sample from it.
            latents = self.sample_from_encoder_outputs(
                self.vae(images, training=False)
            )
            # Know more about the magic number here:
            # https://keras.io/examples/generative/fine_tune_via_textual_inversion/
            latents = latents * 0.18215
            # Sample noise that we'll add to the latents.
            noise = tf.random.normal(tf.shape(latents))
            # Sample a random timestep for each image.
            timesteps = tf.experimental.numpy.random.randint(
                0, self.noise_scheduler.train_timesteps, (batch_size,)
            )
            # Add noise to the latents according to the noise magnitude
            # at each timestep (this is the forward diffusion process).
            noisy_latents = self.noise_scheduler.add_noise(
                tf.cast(latents, noise.dtype), noise, timesteps
            )
            # Get the target for loss depending on the prediction type
            # just the sampled noise for now.
            target = noise  # noise_schedule.predict_epsilon == True
            # Predict the noise residual and compute loss.
            timestep_embedding = tf.map_fn(
                lambda t: self.get_timestep_embedding(t),
                timesteps,
                fn_output_signature=tf.float32,
            )
            timestep_embedding = tf.squeeze(timestep_embedding, 1)
            model_pred = self.diffusion_model(
                [noisy_latents, timestep_embedding, encoded_text],
                training=True,
            )
            loss = self.compiled_loss(target, model_pred)
            if self.use_mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Update parameters of the diffusion model.
        trainable_vars = self.diffusion_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        if self.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients = [tf.clip_by_norm(g, self.max_grad_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {m.name: m.result() for m in self.metrics}

    @staticmethod
    def get_timestep_embedding(timestep, dim=320, max_period=10000):
        half = dim // 2
        log_max_preiod = tf.math.log(tf.cast(max_period, tf.float32))
        freqs = tf.math.exp(
            -log_max_preiod * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return embedding

    @staticmethod
    def sample_from_encoder_outputs(encoder_outputs):
        mean, logvar = tf.split(encoder_outputs, 2, axis=-1)
        logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        std = tf.exp(0.5 * logvar)
        sample = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + std * sample

    def save_weights(
        self, filepath, overwrite=True, save_format=None, options=None
    ):
        # Overriding this method will allow us to use the `ModelCheckpoint`
        # callback directly with this trainer class. In this case, it will
        # only save checkpoint for the `diffusion_model` as since that's what
        # we're training during fine-tuning.
        self.diffusion_model.save_weights(
            filepath=filepath,
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )


# Initialize the trainer and compile it

# Enable mixed-precision training if the underlying GPU has tensor cores.
USE_MP = True
if USE_MP:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = keras_cv.models.stable_diffusion.ImageEncoder(RESOLUTION)
diffusion_ft_trainer = Trainer(
    diffusion_model=keras_cv.models.stable_diffusion.DiffusionModel(
        RESOLUTION, RESOLUTION, MAX_PROMPT_LENGTH
    ),
    # Remove the top layer from the encoder, which cuts off the variance
    # and only returns the mean.
    vae=tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-2].output,
    ),
    noise_scheduler=keras_cv.models.stable_diffusion.NoiseScheduler(),
    use_mixed_precision=USE_MP,
)
# Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5
# 136824240/136824240 [==============================] - 9s 0us/step
# Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5
# 3439090152/3439090152 [==============================] - 15s 0us/step

# These hyperparameters come from this tutorial by Hugging Face:
# https://huggingface.co/docs/diffusers/training/text2image
lr = 1e-5
beta_1, beta_2 = 0.9, 0.999
weight_decay = (1e-2,)
epsilon = 1e-08

optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)
diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")


# Fine-tuning

epochs = 2
ckpt_path = "finetuned_stable_diffusion.h5"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)
diffusion_ft_trainer.fit(
    training_dataset, epochs=epochs, callbacks=[ckpt_callback]
)
# Epoch 1/2
# 209/209 [==============================] - 92s 433ms/step - loss: 0.0634
# Epoch 2/2
# 209/209 [==============================] - 90s 423ms/step - loss: 0.0726
# <keras.callbacks.History at 0x7a126f03e230>


# Inference

weights_path = tf.keras.utils.get_file(
    origin="https://huggingface.co/sayakpaul/kerascv_sd_pokemon_finetuned/"
           "resolve/main/ckpt_epochs_72_res_512_mp_True.h5"
)

img_height = img_width = 512
pokemon_model = keras_cv.models.StableDiffusion(
    img_width=img_width, img_height=img_height
)
# We just reload the weights of the fine-tuned diffusion model.
pokemon_model.diffusion_model.load_weights(weights_path)
# Downloading data from ...
# 3439089408/3439089408 [==============================] - 16s 0us/step


prompts = ["Yoda", "Hello Kitty", "A pokemon with red eyes"]
images_to_generate = 3
outputs = {}

for prompt in prompts:
    generated_images = pokemon_model.text_to_image(
        prompt, batch_size=images_to_generate, unconditional_guidance_scale=40
    )
    outputs.update({prompt: generated_images})
# 50/50 [==============================] - 25s 230ms/step
# Downloading data from https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5
# 198180272/198180272 [==============================] - 1s 0us/step
# 50/50 [==============================] - 12s 230ms/step
# 50/50 [==============================] - 11s 229ms/step


def plot_images(images, title):
    plt.figure(figsize=(20, 20))
    for _i in range(len(images)):
        _ax = plt.subplot(1, len(images), _i + 1)
        plt.imshow(images[_i])
        plt.title(title, fontsize=12)
        plt.axis("off")


for prompt in outputs:
    plot_images(outputs[prompt], prompt)
