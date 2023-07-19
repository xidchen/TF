import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import time

tf.keras.utils.set_random_seed(seed=0)


# Introduction

model = keras_cv.models.StableDiffusion(jit_compile=True)


def plot_images(imgs):
    plt.figure(figsize=(20, 20))
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


images = model.text_to_image(
    prompt="photograph of carey mulligan, with a young and inmature smile, "
           "with her full body in a sexy nightie inside and "
           "a mature and elegant outfit outside. the whole body in the image",
    batch_size=3,
)
plot_images(images)

images = model.text_to_image(
    prompt="a chinese 24-year-old woman, looking like carey mulligan, "
           "black hair, standing on 2023 shanghai huaihai road, "
           "hyper realistic portrait photography, pale skin, dress, "
           "wide shot, naturallighting, kodak portra 800",
    batch_size=3,
)
plot_images(images)

images = model.text_to_image(
    prompt="realistic/distinct facial features, macro photo shot of "
           "a beautiful asian girl looking up at sky, with stars sparkling "
           "in her eyes and her eyelashes covered in water droplets, "
           "with skin pure white, hair shining in the sunlight, "
           "hyperrealistic, colorful cinematic lighting, in the style of "
           "realistic and hyper-detailed renderings, contoured shading, "
           "extreme iridescent reflection overexposure",
    batch_size=3,
)
plot_images(images)


# Perks of KerasCV

benchmark_result = []
start = time.time()
images = model.text_to_image(
    prompt="A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Standard", end - start])
plot_images(images)

print(f"Standard model: {(end - start):.2f} seconds")
tf.keras.backend.clear_session()   # Clear session to preserve memory.


# Mixed precision

tf.keras.mixed_precision.set_global_policy("mixed_float16")

model = keras_cv.models.StableDiffusion()

print(f"Compute dtype: {model.diffusion_model.compute_dtype}")
print(f"Variable dtype: {model.diffusion_model.variable_dtype}")

# Warm up model to run graph tracing before benchmarking.
model.text_to_image("warming up the model", batch_size=3)

start = time.time()
images = model.text_to_image(
    prompt="a cute magical flying dog, fantasy art, golden color, "
           "high quality, highly detailed, elegant, sharp focus, concept art, "
           "character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Mixed Precision", end - start])
plot_images(images)

print(f"Mixed precision model: {(end - start):.2f} seconds")
tf.keras.backend.clear_session()


# XLA Compilation

# Set back to the default for benchmarking purposes.
tf.keras.mixed_precision.set_global_policy("float32")

model = keras_cv.models.StableDiffusion(jit_compile=True)
# Before we benchmark the model, we run inference once
# to make sure the TensorFlow graph has already been traced.
images = model.text_to_image(prompt="An avocado armchair", batch_size=3)
plot_images(images)

start = time.time()
images = model.text_to_image(
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["XLA", end - start])
plot_images(images)

print(f"With XLA: {(end - start):.2f} seconds")
tf.keras.backend.clear_session()


# Putting it all together

tf.keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(jit_compile=True)

# Let's make sure to warm up the model
images = model.text_to_image(
    prompt="Teddy bears conducting machine learning research",
    batch_size=3,
)
plot_images(images)

start = time.time()
images = model.text_to_image(
    prompt="A mysterious dark stranger visits the great pyramids of egypt, "
           "high quality, highly detailed, elegant, sharp focus, "
           "concept art, character concepts, digital painting",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["XLA + Mixed Precision", end - start])
plot_images(images)

print(f"XLA + mixed precision: {(end - start):.2f} seconds")

print("{:<20} {:<20}".format("Model", "Runtime"))
for result in benchmark_result:
    name, runtime = result
    print("{:<20} {:<20}".format(name, runtime))


# Model                Runtime
# Standard             15.445802211761475
# Mixed Precision      11.76381802558899
# XLA                  11.772337198257446
# XLA + Mixed Precision 7.521973609924316
