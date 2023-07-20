import keras_cv
import math
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.keras.utils.set_random_seed(seed=0)


model = keras_cv.models.StableDiffusion(jit_compile=True)


# Interpolating between text prompts

prompt_1 = 'A watercolor painting of a Golden Retriever at the beach'
prompt_2 = 'A still life DSLR photo of a bowl of fruit'
interpolation_steps = 5

encoding_1 = tf.squeeze(model.encode_text(prompt_1))
encoding_2 = tf.squeeze(model.encode_text(prompt_2))
interpolated_encodings = tf.linspace(
    start=encoding_1,
    stop=encoding_2,
    num=interpolation_steps
)
print(f'Encoding shapes: {encoding_1.shape}')

noise = tf.random.normal(shape=(512 // 8, 512 // 8, 4))
images = model.generate_image(
    encoded_text=interpolated_encodings,
    batch_size=interpolation_steps,
    diffusion_noise=noise
)


def export_as_gif(filename, imgs, frames_per_second=10, rubber_band=False):
    if rubber_band:
        imgs += imgs[2:-1][::-1]
    imgs[0].save(
        filename,
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // frames_per_second,
        loop=0,
    )


export_as_gif(
    "doggo-and-fruit-5.gif",
    [PIL.Image.fromarray(img) for img in images],
    frames_per_second=2,
    rubber_band=True,
)


interpolation_steps = 150
batch_size = 3
batches = interpolation_steps // batch_size

interpolated_encodings = tf.linspace(
    start=encoding_1,
    stop=encoding_2,
    num=interpolation_steps
)
batched_encodings = tf.split(
    value=interpolated_encodings,
    num_or_size_splits=batches
)

images = []
for batch in range(batches):
    images += [
        PIL.Image.fromarray(img)
        for img in model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=noise,
        )
    ]

export_as_gif("doggo-and-fruit-150.gif", images, rubber_band=True)

prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
prompt_3 = "The eiffel tower in the style of starry night"
prompt_4 = "An architectural sketch of a skyscraper"

interpolation_steps = 6
batch_size = 3
batches = interpolation_steps ** 2 // batch_size

encoding_1 = tf.squeeze(model.encode_text(prompt_1))
encoding_2 = tf.squeeze(model.encode_text(prompt_2))
encoding_3 = tf.squeeze(model.encode_text(prompt_3))
encoding_4 = tf.squeeze(model.encode_text(prompt_4))

interpolated_encodings = tf.linspace(
    start=tf.linspace(encoding_1, encoding_2, interpolation_steps),
    stop=tf.linspace(encoding_3, encoding_4, interpolation_steps),
    num=interpolation_steps,
)
interpolated_encodings = tf.reshape(
    interpolated_encodings, shape=(interpolation_steps ** 2, 77, 768)
)
batched_encodings = tf.split(interpolated_encodings, batches)

images = []
for batch in range(batches):
    images.append(
        model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            diffusion_noise=noise,
        )
    )


def plot_grid(imgs, path, grid_size, scale=2):
    fig = plt.figure(figsize=(grid_size * scale, grid_size * scale))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.axis("off")
    imgs = imgs.astype(int)
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            plt.subplot(grid_size, grid_size, index + 1)
            plt.imshow(imgs[index].astype("uint8"))
            plt.axis("off")
            plt.margins(x=0, y=0)
    plt.savefig(
        fname=path,
        pad_inches=0,
        bbox_inches="tight",
        transparent=False,
        dpi=60,
    )


images = np.concatenate(images)
plot_grid(images, "4-way-interpolation.jpg", interpolation_steps)


images = []
for batch in range(batches):
    images.append(
        model.generate_image(batched_encodings[batch], batch_size=batch_size)
    )

images = np.concatenate(images)
plot_grid(images, "4-way-interpolation-varying-noise.jpg", interpolation_steps)


# A walk around a text prompt

walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size
step_size = 0.005

encoding = tf.squeeze(
    model.encode_text("The Eiffel Tower in the style of starry night")
)
# Note that (77, 768) is the shape of the text encoding.
delta = tf.ones_like(encoding) * step_size

walked_encodings = []
for step_index in range(walk_steps):
    walked_encodings.append(encoding)
    encoding += delta
walked_encodings = tf.stack(walked_encodings)
batched_encodings = tf.split(walked_encodings, batches)

images = []
for batch in range(batches):
    images += [
        PIL.Image.fromarray(img)
        for img in model.generate_image(
            batched_encodings[batch],
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=noise,
        )
    ]

export_as_gif("eiffel-tower-starry-night.gif", images, rubber_band=True)


# A circular walk through the diffusion noise space for a single prompt

prompt = "An oil paintings of cows in a field next to a windmill in Holland"
encoding = tf.squeeze(model.encode_text(prompt))
walk_steps = 150
batch_size = 3
batches = walk_steps // batch_size

walk_noise_x = tf.random.normal(noise.shape, dtype=tf.float64)
walk_noise_y = tf.random.normal(noise.shape, dtype=tf.float64)

walk_scale_x = tf.cos(tf.linspace(0, 2, walk_steps) * math.pi)
walk_scale_y = tf.sin(tf.linspace(0, 2, walk_steps) * math.pi)
noise_x = tf.tensordot(walk_scale_x, walk_noise_x, axes=0)
noise_y = tf.tensordot(walk_scale_y, walk_noise_y, axes=0)
noise = tf.add(noise_x, noise_y)
batched_noise = tf.split(noise, batches)

images = []
for batch in range(batches):
    images += [
        PIL.Image.fromarray(img)
        for img in model.generate_image(
            encoding,
            batch_size=batch_size,
            num_steps=25,
            diffusion_noise=batched_noise[batch],
        )
    ]

export_as_gif("cows.gif", images)
