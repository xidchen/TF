import cv2
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import shutil
import tensorflow as tf

# Prepare dataset

path_images = "/101_ObjectCategories/airplanes/"
path_annot = "/Annotations/Airplanes_Side_2/"

path_to_downloaded_file = tf.keras.utils.get_file(
    fname="caltech_101_zipped",
    origin="https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip",
    extract=True,
    archive_format="zip",  # downloaded file format
    cache_dir="/",  # cache and extract in current directory
)

shutil.unpack_archive("/datasets/caltech-101/101_ObjectCategories.tar.gz", "/")
shutil.unpack_archive("/datasets/caltech-101/Annotations.tar", "/")

image_paths = [
    f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))
]
annot_paths = [
    f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))
]

image_paths.sort()
annot_paths.sort()

IMAGE_SIZE = 224

images, targets = [], []

# loop over the annotations and images, preprocess them and store in lists
for i in range(0, len(annot_paths)):
    # Access bounding box coordinates
    annot = scipy.io.loadmat(path_annot + annot_paths[i])["box_coord"][0]

    top_left_x, top_left_y = annot[2], annot[0]
    bottom_right_x, bottom_right_y = annot[3], annot[1]

    image = tf.keras.utils.load_img(
        path_images + image_paths[i],
    )
    w, h = image.size[:2]

    # resize train set images
    if i < int(len(annot_paths) * 0.8):
        # resize image if it is for training dataset
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # convert image to array and append to list
    images.append(tf.keras.utils.img_to_array(image))

    # apply relative scaling to bounding boxes as per given image and append to list
    targets.append(
        (
            float(top_left_x) / w,
            float(top_left_y) / h,
            float(bottom_right_x) / w,
            float(bottom_right_y) / h,
        )
    )

# Convert the list to numpy array, split to train and test dataset
x_train, y_train = (
    np.asarray(images[: int(len(images) * 0.8)]),
    np.asarray(targets[: int(len(targets) * 0.8)]),
)
x_test, y_test = (
    np.asarray(images[int(len(images) * 0.8):], dtype=object),
    np.asarray(targets[int(len(targets) * 0.8):], dtype=object),
)


# Implement multilayer-perceptron (MLP)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


# Implement the patch creation layer

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": INPUT_SHAPE,
                "patch_size": PATCH_SIZE,
                "num_patches": NUM_PATCHES,
                "projection_dim": PROJECTION_DIM,
                "num_heads": NUM_HEADS,
                "transformer_units": TRANSFORMER_UNITS,
                "transformer_layers": TRANSFOMRER_LAYERS,
                "mlp_head_units": MLP_HEAD_UNITS,
            }
        )
        return config

    def call(self, _images, *args):
        batch_size = tf.shape(_images)[0]
        _patches = tf.image.extract_patches(
            images=_images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # return patches
        return tf.reshape(_patches, [batch_size, -1, _patches.shape[-1]])


# Display patches for an input image

PATCH_SIZE = 32  # Size of the patches to be extracted from the input images

plt.figure(figsize=(4, 4))
plt.imshow(x_train[0].astype("uint8"))
plt.axis("off")

patches = Patches(PATCH_SIZE)(tf.convert_to_tensor([x_train[0]]))
print(f"Image size: {IMAGE_SIZE} X {IMAGE_SIZE}")
print(f"Patch size: {PATCH_SIZE} X {PATCH_SIZE}")
print(f"{patches.shape[1]} patches per image")
print(f"{patches.shape[-1]} elements per patch")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")


# Implement the patch encoding layer

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": INPUT_SHAPE,
                "patch_size": PATCH_SIZE,
                "num_patches": NUM_PATCHES,
                "projection_dim": PROJECTION_DIM,
                "num_heads": NUM_HEADS,
                "transformer_units": TRANSFORMER_UNITS,
                "transformer_layers": TRANSFOMRER_LAYERS,
                "mlp_head_units": MLP_HEAD_UNITS,
            }
        )
        return config

    def call(self, _patch, *args):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(_patch) + self.position_embedding(positions)
        return encoded


# Build the ViT model

def create_vit_object_detector(
        input_shape,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units,
):
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Create patches
    _patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(_patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = tf.keras.layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return tf.keras.Model(inputs=inputs, outputs=bounding_box)


# Run the experiment

def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):
    optimizer = tf.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

    checkpoint_filepath = "logs/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history


INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)  # input image shape
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
# Size of the transformer layers
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
TRANSFOMRER_LAYERS = 4
MLP_HEAD_UNITS = [2048, 1024, 512, 64, 32]  # Size of the dense layers

vit_object_detector = create_vit_object_detector(
    INPUT_SHAPE,
    PATCH_SIZE,
    NUM_PATCHES,
    PROJECTION_DIM,
    NUM_HEADS,
    TRANSFORMER_UNITS,
    TRANSFOMRER_LAYERS,
    MLP_HEAD_UNITS,
)

# Train model
model_history = run_experiment(
    vit_object_detector, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, NUM_EPOCHS
)

# Evaluate the model

# Saves the model in current path
vit_object_detector.save("vit_object_detector.h5", save_format="h5")


# To calculate IoU (intersection over union, given two bounding boxes)
def bounding_box_intersection_over_union(_box_predicted, _box_truth):
    # get (x, y) coordinates of intersection of bounding boxes
    top_x_intersect = max(_box_predicted[0], _box_truth[0])
    top_y_intersect = max(_box_predicted[1], _box_truth[1])
    bottom_x_intersect = min(_box_predicted[2], _box_truth[2])
    bottom_y_intersect = min(_box_predicted[3], _box_truth[3])

    # calculate area of the intersection bb (bounding box)
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    )

    # calculate area of the prediction bb and ground-truth bb
    box_predicted_area = (_box_predicted[2] - _box_predicted[0] + 1) * (
            _box_predicted[3] - _box_predicted[1] + 1
    )
    box_truth_area = (_box_truth[2] - _box_truth[0] + 1) * (
            _box_truth[3] - _box_truth[1] + 1
    )

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb and ground truth
    # bb areas subtracted by  the interesection area

    # return ioU
    return intersection_area / float(
        box_predicted_area + box_truth_area - intersection_area
    )


i, mean_iou = 0, 0

# Compare results for 10 images in the test set
for input_image in x_test[:10]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = input_image

    # Display the image
    ax1.imshow(im.astype("uint8"))
    ax2.imshow(im.astype("uint8"))

    input_image = cv2.resize(
        input_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )
    input_image = np.expand_dims(input_image, axis=0)
    preds = vit_object_detector.predict(input_image)[0]

    (h, w) = im.shape[0:2]

    top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)

    bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    # Create the bounding box
    rect = pch.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax1.add_patch(rect)
    ax1.set_xlabel(
        "Predicted: "
        + str(top_left_x)
        + ", "
        + str(top_left_y)
        + ", "
        + str(bottom_right_x)
        + ", "
        + str(bottom_right_y)
    )

    top_left_x, top_left_y = int(y_test[i][0] * w), int(y_test[i][1] * h)

    bottom_right_x, bottom_right_y = int(y_test[i][2] * w), int(y_test[i][3] * h)

    box_truth = top_left_x, top_left_y, bottom_right_x, bottom_right_y

    mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
    # Create the bounding box
    rect = pch.Rectangle(
        (top_left_x, top_left_y),
        bottom_right_x - top_left_x,
        bottom_right_y - top_left_y,
        facecolor="none",
        edgecolor="red",
        linewidth=1,
    )
    # Add the bounding box to the image
    ax2.add_patch(rect)
    ax2.set_xlabel(
        "Target: "
        + str(top_left_x)
        + ", "
        + str(top_left_y)
        + ", "
        + str(bottom_right_x)
        + ", "
        + str(bottom_right_y)
        + "\n"
        + "IoU"
        + str(bounding_box_intersection_over_union(box_predicted, box_truth))
    )
    i = i + 1

print("mean_iou: " + str(mean_iou / len(x_test[:10])))
plt.show()
