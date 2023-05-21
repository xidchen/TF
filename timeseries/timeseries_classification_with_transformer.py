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

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

n_classes = len(np.unique(y_train))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


# Build the model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate):
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout_rate
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout_rate,
    mlp_dropout_rate,
):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(mlp_dropout_rate)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


# Train and evaluate

model = build_model(
    input_shape=x_train.shape[1:],
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout_rate=0.4,
    dropout_rate=0.25,
)

epochs = 500
batch_size = 32
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model_trf.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=20, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
]
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
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

model = tf.keras.models.load_model('best_model_trf.h5')
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
