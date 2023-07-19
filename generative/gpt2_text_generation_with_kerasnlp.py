import json
import keras_nlp
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import time


# Load a pre-trained GPT-2 model and generate some text
# (gpt2_large is much better than gpt2_base)

# To speed up training and generation, we use preprocessor of length 128
# instead of full length 1024.
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    "gpt2_large_en",
    sequence_length=128,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    "gpt2_large_en",
    preprocessor=preprocessor,
)

start = time.time()
output = gpt2_lm.generate(inputs="My trip to Yosemite was", max_length=200)
print(f"\nGPT-2 output:\n{output}")
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

start = time.time()
output = gpt2_lm.generate(inputs="The Italian restaurant is", max_length=200)
print(f"\nGPT-2 output:\n{output}")
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")


# Finetune on Reddit dataset

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
print(f"length of reddit_tifu dataset: {reddit_ds.cardinality()}\n")
for document, title in reddit_ds.take(5).shuffle(buffer_size=5):
    print(document.numpy())
    print(title.numpy())
    print()
print()
train_ds = (
    reddit_ds
    .map(lambda doc, _: doc)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-5,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
gpt2_lm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    weighted_metrics=["accuracy"],
)
gpt2_lm.fit(train_ds, epochs=num_epochs)

start = time.time()
output = gpt2_lm.generate(inputs="I like basketball", max_length=200)
print(f"\nGPT-2 output:\n{output}")
end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")


# Into the Sampling Method

# Use a string identifier.
gpt2_lm.compile(sampler="top_k")  # default sampler
output = gpt2_lm.generate(inputs="I like basketball", max_length=200)
print(f"GPT-2 output (top-k):\n{output}\n")

# Use a `Sampler` instance. `GreedySampler` tends to repeat itself.
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)
output = gpt2_lm.generate(inputs="I like basketball", max_length=200)
print(f"GPT-2 output (greedy):\n{output}\n")


# Finetune on Chinese Poem Dataset

# !git clone https://github.com/chinese-poetry/chinese-poetry.git
poem_collection = []
for file in os.listdir("chinese-poetry/宋词"):
    if ".json" not in file or "ci" not in file:
        continue
    full_filename = f"chinese-poetry/宋词/{file}"
    with open(full_filename, "r") as f:
        content = json.load(f)
        poem_collection.extend(content)
paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]
print(f"Length of dataset: {len(paragraphs)}\n")
for p in paragraphs[88:100]:
    print(p)
print()

train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Running through the whole dataset takes long, only take `500`
# and run 1 epoch for demo purposes.
train_ds = train_ds.take(4000)
# num_epochs = 1
num_epochs = 4

learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=5e-4,
    decay_steps=train_ds.cardinality() * num_epochs,
    end_learning_rate=0.0,
)
gpt2_lm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    weighted_metrics=["accuracy"],
)
gpt2_lm.fit(train_ds, epochs=num_epochs)
print()

output = gpt2_lm.generate("北国风光", max_length=200)
print(f"{output}\n")
output = gpt2_lm.generate("大江东去", max_length=200)
print(f"{output}\n")
