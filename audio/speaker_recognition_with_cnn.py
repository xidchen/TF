import numpy as np
import os
import shutil
import tensorflow as tf


# Setup

ROOT_DIR = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
DATASET_ROOT = os.path.join(ROOT_DIR, "16000_pcm_speeches")

AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000
SCALE = 0.5
BATCH_SIZE = 128
EPOCHS = 100


# Data preparation

if os.path.exists(DATASET_AUDIO_PATH) is False:
    os.makedirs(DATASET_AUDIO_PATH)
if os.path.exists(DATASET_NOISE_PATH) is False:
    os.makedirs(DATASET_NOISE_PATH)
for folder in os.listdir(DATASET_ROOT):
    if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
        if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
            continue
        elif folder in ["other", "_background_noise_"]:
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_NOISE_PATH, folder),
            )
        else:
            shutil.move(
                os.path.join(DATASET_ROOT, folder),
                os.path.join(DATASET_AUDIO_PATH, folder),
            )


# Noise preparation

noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH):
    subdir_path = os.path.join(DATASET_NOISE_PATH, subdir)
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

print(
    f"Found {len(noise_paths)} files belonging "
    f"to {len(os.listdir(DATASET_NOISE_PATH))} directories"
)

command = (
    "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
    "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)

os.system(command)


def load_noise_sample(arg_path):
    """
    Split noise into chunks of 16000 each
    """
    _sample, _sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(arg_path), desired_channels=1
    )
    if _sampling_rate == SAMPLING_RATE:
        slices = int(_sample.shape[0] / SAMPLING_RATE)
        _sample = tf.split(_sample[:slices * SAMPLING_RATE], slices)
        return _sample
    else:
        print(f"Sampling rate for {arg_path} is incorrect. Ignoring it.")
        return None


noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)


print(
    f"{len(noise_paths)} noise files were split into "
    f"{noises.shape[0]} noise samples where each is "
    f"{noises.shape[1] // SAMPLING_RATE} second(s) long"
)


# Dataset generation

def path_to_audio(arg_path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(arg_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    return audio


def paths_and_labels_to_dataset(arg_audio_paths, arg_labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(arg_audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(arg_labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def add_noise(arg_audio, arg_noises=None, arg_scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(arg_audio)[0],), 0, arg_noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)
        prop = (
                tf.reduce_max(arg_audio, axis=1) / tf.reduce_max(noise, axis=1)
        )
        prop = (
            tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(arg_audio)[1], axis=1)
        )
        return arg_audio + noise * prop * arg_scale


def audio_to_fft(arg_audio):
    """
    Since tf.signal.fft applies FFT on the innermost dimension,
    we need to squeeze the dimensions and then expand them again after FFT
    """
    arg_audio = tf.squeeze(arg_audio, axis=-1)
    fft = tf.signal.fft(tf.cast(
        tf.complex(real=arg_audio, imag=tf.zeros_like(arg_audio)), tf.complex64
    ))
    fft = tf.expand_dims(fft, axis=-1)
    return tf.abs(fft[:, :(arg_audio.shape[1] // 2), :])


class_names = os.listdir(DATASET_AUDIO_PATH)
print(f"Our class names: {class_names}")

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print(f"Processing speaker {name}")
    dir_path = os.path.join(DATASET_AUDIO_PATH, name)
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(f"Found {len(audio_paths)} files belonging to {len(class_names)} classes.")

rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

num_val_samples = int(VALID_SPLIT * len(audio_paths))
print(f"Using {len(audio_paths) - num_val_samples} files for training.")
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print(f"Using {num_val_samples} files for validation.")
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED)
train_ds = train_ds.batch(batch_size=BATCH_SIZE)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED)
valid_ds = valid_ds.batch(batch_size=32)

train_ds = train_ds.map(
    map_func=lambda x, y: (add_noise(x, noises, arg_scale=SCALE), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.map(
    map_func=lambda x, y: (audio_to_fft(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    map_func=lambda x, y: (tf.reshape(x, shape=[-1, SAMPLING_RATE, 1]), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_ds = valid_ds.map(
    map_func=lambda x, y: (audio_to_fft(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
valid_ds = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# Model Definition

def residual_block(x, filters, conv_num=3, activation="relu"):
    s = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="same")(x)
    for i in range(conv_num - 1):
        x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.Add()([x, s])
    x = tf.keras.layers.Activation(activation)(x)
    return tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = residual_block(inputs, filters=16, conv_num=2)
    x = residual_block(x, filters=32, conv_num=2)
    x = residual_block(x, filters=64, conv_num=3)
    x = residual_block(x, filters=128, conv_num=3)
    x = residual_block(x, filters=128, conv_num=3)
    x = tf.keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(
        units=num_classes, activation="softmax", name="output"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


model = build_model(
    input_shape=(SAMPLING_RATE // 2, 1), num_classes=len(class_names)
)

model.summary()

model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model_save_filename = "model.h5"

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)
model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_filename, monitor="val_accuracy", save_best_only=True
)


# Training

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    validation_data=valid_ds
)


# Evaluation

print(model.evaluate(valid_ds))


# Demonstration

SAMPLES_TO_DISPLAY = 10

test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED)
test_ds = test_ds.batch(BATCH_SIZE)

test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, arg_scale=SCALE), y))

for audios, labels in test_ds.take(1):
    ffts = audio_to_fft(audios)
    y_pred = model.predict(ffts)
    rnd = np.random.randint(low=0, high=BATCH_SIZE, size=SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]
    for index in range(SAMPLES_TO_DISPLAY):
        print(
            f"Speaker: {class_names[labels[index]]} - "
            f"Predicted: {class_names[y_pred[index]]}"
        )
