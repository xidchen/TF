# Set up
import abc
import tensorflow as tf


# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')

# Use tf.data to batch and shuffle the dataset
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(tf.keras.Model, abc.ABC):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, inputs, **kwargs):
        return self.d2(self.d1(self.flatten(self.conv1(inputs))))


# Create an instance of the model
model = MyModel()

# Choose an optimizer and loss function for training
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model
train_loss = tf.metrics.Mean(name='train_loss')
train_accuracy = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.metrics.Mean(name='test_loss')
test_accuracy = tf.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Use tf.GradientTape to train the model
@tf.function
def train_step(_images, _labels):
    with tf.GradientTape() as tape:
        _predictions = model(_images, training=True)
        _loss = loss(_labels, _predictions)
    gradients = tape.gradient(_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(_loss)
    train_accuracy(_labels, _predictions)


# Test the model
@tf.function
def test_step(_images, _labels):
    _predictions = model(_images, training=False)
    _loss = loss(_labels, _predictions)

    test_loss(_loss)
    test_accuracy(_labels, _predictions)


EPOCHS = 2

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for images, labels in test_ds:
        test_step(images, labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result()}'
    )
