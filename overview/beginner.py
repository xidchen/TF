# Set up
import tensorflow as tf


# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Define a loss function for training
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# Use the Model.fit method to adjust model parameters and minimize the loss
model.fit(x_train, y_train, epochs=5)

# The Model.evaluate method checks the models performance
model.evaluate(x_test, y_test, verbose=2)

# If you want your model to return a probability,
# you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
