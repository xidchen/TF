import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(seed=0)


# Load the data: the FordA dataset

def readucr(filename):
    data = np.loadtxt(filename, delimiter='\t')
    x = data[:, 1:]
    y = data[:, 0]
    return x, y.astype(int)


root_url = 'https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'

x_train, y_train = readucr(root_url + 'FordA_TRAIN.tsv')
x_test, y_test = readucr(root_url + 'FordA_TEST.tsv')


# Visualize the data

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label='class ' + str(c))
plt.legend(loc='best')
plt.show()


# Standardize the data

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


# Build a model

def make_model(input_shape):
    input_layer = tf.keras.Input(input_shape)
    conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    gap = tf.keras.layers.GlobalAvgPool1D()(conv3)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(gap)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])


# Train the model

epochs = 500
batch_size = 32
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model_cnn.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-4),
    tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True),
]
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.metrics.SparseCategoricalAccuracy()],
)
model.summary()
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_split=0.2,
)


# Evaluate model on test data

model = tf.keras.models.load_model('best_model_cnn.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
print(f'Test loss: {test_loss}')


# Plot the model's training and validation loss

metric = 'sparse_categorical_accuracy'
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history['val_' + metric])
plt.title('model ' + metric)
plt.xlabel('epoch', fontsize='large')
plt.ylabel(metric, fontsize='large')
plt.legend(['train', 'val'], loc='best')
plt.show()
