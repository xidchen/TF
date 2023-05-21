import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.model_selection as sms
import sklearn.preprocessing as sp
import tensorflow as tf

tf.keras.utils.set_random_seed(seed=0)


ROOT_DIR = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
DATA_PATH = os.path.join(ROOT_DIR, 'eeg-data.csv')
QUALITY_THRESHOLD = 128
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2


if not os.path.exists(DATA_PATH):
    gdown.download(
        url='https://drive.google.com/uc?id=1V5B7Bt6aJm0UHbR7cRKBEK8jx7lYPVuX',
        output=DATA_PATH,
    )


# Read data from eeg-data.csv

eeg = pd.read_csv(DATA_PATH)
unlabeled_eeg = eeg[eeg['label'] == 'unlabeled']
eeg = eeg.loc[eeg['label'] != 'unlabeled']
eeg = eeg.loc[eeg['label'] != 'everyone paired']
eeg.drop(
    [
        'indra_time',
        'Unnamed: 0',
        'browser_latency',
        'reading_time',
        'attention_esense',
        'meditation_esense',
        'updatedAt',
        'createdAt',
    ],
    axis=1,
    inplace=True,
)
eeg.reset_index(drop=True, inplace=True)


def convert_string_data_to_values(value_string):
    return eval(value_string)


eeg['raw_values'] = eeg['raw_values'].apply(convert_string_data_to_values)

eeg = eeg.loc[eeg['signal_quality'] < QUALITY_THRESHOLD]


# Visualize one random sample from the data

def view_eeg_plot(idx):
    data = eeg.loc[idx, 'raw_values']
    plt.plot(data)
    plt.title(f'Sample random plot')
    plt.show()


# Pre-process and collate data

print('Before replacing labels')
print(eeg['label'].unique())
print(len(eeg['label'].unique()), '\n')

eeg.replace(
    {
        'label': {
            'blink1': 'blink',
            'blink2': 'blink',
            'blink3': 'blink',
            'blink4': 'blink',
            'blink5': 'blink',
            'math1': 'math',
            'math2': 'math',
            'math3': 'math',
            'math4': 'math',
            'math5': 'math',
            'math6': 'math',
            'math7': 'math',
            'math8': 'math',
            'math9': 'math',
            'math10': 'math',
            'math11': 'math',
            'math12': 'math',
            'thinkOfItems-ver1': 'thinkOfItems',
            'thinkOfItems-ver2': 'thinkOfItems',
            'video-ver1': 'video',
            'video-ver2': 'video',
            'thinkOfItemsInstruction-ver1': 'thinkOfItemsInstruction',
            'thinkOfItemsInstruction-ver2': 'thinkOfItemsInstruction',
            'colorRound1-1': 'colorRound1',
            'colorRound1-2': 'colorRound1',
            'colorRound1-3': 'colorRound1',
            'colorRound1-4': 'colorRound1',
            'colorRound1-5': 'colorRound1',
            'colorRound1-6': 'colorRound1',
            'colorRound2-1': 'colorRound2',
            'colorRound2-2': 'colorRound2',
            'colorRound2-3': 'colorRound2',
            'colorRound2-4': 'colorRound2',
            'colorRound2-5': 'colorRound2',
            'colorRound2-6': 'colorRound2',
            'colorRound3-1': 'colorRound3',
            'colorRound3-2': 'colorRound3',
            'colorRound3-3': 'colorRound3',
            'colorRound3-4': 'colorRound3',
            'colorRound3-5': 'colorRound3',
            'colorRound3-6': 'colorRound3',
            'colorRound4-1': 'colorRound4',
            'colorRound4-2': 'colorRound4',
            'colorRound4-3': 'colorRound4',
            'colorRound4-4': 'colorRound4',
            'colorRound4-5': 'colorRound4',
            'colorRound4-6': 'colorRound4',
            'colorRound5-1': 'colorRound5',
            'colorRound5-2': 'colorRound5',
            'colorRound5-3': 'colorRound5',
            'colorRound5-4': 'colorRound5',
            'colorRound5-5': 'colorRound5',
            'colorRound5-6': 'colorRound5',
            'colorInstruction1': 'colorInstruction',
            'colorInstruction2': 'colorInstruction',
            'readyRound1': 'readyRound',
            'readyRound2': 'readyRound',
            'readyRound3': 'readyRound',
            'readyRound4': 'readyRound',
            'readyRound5': 'readyRound',
            'colorRound1': 'colorRound',
            'colorRound2': 'colorRound',
            'colorRound3': 'colorRound',
            'colorRound4': 'colorRound',
            'colorRound5': 'colorRound',
        }
    },
    inplace=True,
)

print('After replacing labels')
print(eeg['label'].unique())
print(len(eeg['label'].unique()), '\n')

le = sp.LabelEncoder()
le.fit(eeg['label'])
eeg['label'] = le.transform(eeg['label'])

num_classes = len(eeg['label'].unique())
print(f'Number of classes: {num_classes}')

plt.bar(range(num_classes), height=eeg['label'].value_counts())
plt.title('Number of samples per class')
plt.show()


# Scale and split data

scaler = sp.MinMaxScaler()
series_list = [
    scaler.fit_transform(np.asarray(i).reshape((-1, 1)))
    for i in eeg['raw_values']
]
labels_list = [i for i in eeg['label']]

x_train, x_test, y_train, y_test = sms.train_test_split(
    series_list, labels_list, test_size=0.15, random_state=42, shuffle=True
)

print(
    f'Length of x_train : {len(x_train)}\n'
    f'Length of x_test : {len(x_test)}\n'
    f'Length of y_train : {len(y_train)}\n'
    f'Length of y_test : {len(y_test)}'
)

x_train = np.asarray(x_train).astype(np.float32).reshape((-1, 512, 1))
y_train = np.asarray(y_train).astype(np.float32).reshape((-1, 1))
y_train = tf.keras.utils.to_categorical(y_train)

x_test = np.asarray(x_test).astype(np.float32).reshape((-1, 512, 1))
y_test = np.asarray(y_test).astype(np.float32).reshape((-1, 1))
y_test = tf.keras.utils.to_categorical(y_test)


# Prepare tf.data.Dataset

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# Make Class Weights using Naive method

vals_dict = {}
for i in eeg['label']:
    if i in vals_dict.keys():
        vals_dict[i] += 1
    else:
        vals_dict[i] = 1
total = sum(vals_dict.values())

# Formula used - Naive method where
# weight = 1 - (no. of samples present / total no. of samples)
# So more the samples, lower the weight

weight_dict = {k: (1 - (v / total)) for k, v in vals_dict.items()}
print(weight_dict)


# Define function to plot all the metrics present in a keras.callbacks.History

def plot_history_metrics(history: tf.keras.callbacks.History):
    total_plots = len(history.history)
    cols = total_plots // 2
    rows = total_plots // cols
    if total_plots % cols:
        rows += 1
    pos = range(1, total_plots + 1)
    plt.figure(figsize=(15, 10))
    for i, (key, value) in enumerate(history.history.items()):
        plt.subplot(rows, cols, pos[i])
        plt.plot(range(len(value)), value)
        plt.title(str(key))
    plt.show()


# Define function to generate Convolutional model

def create_model():
    input_layer = tf.keras.Input(shape=(512, 1))
    x = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, strides=2, padding='same', activation='relu'
    )(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, strides=2, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=128, kernel_size=5, strides=2, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=256, kernel_size=5, strides=2, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=512, kernel_size=7, strides=2, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(
        filters=1024, kernel_size=7, strides=2, padding='same', activation='relu'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        2048, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        1024, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2()
    )(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


# Get Model summary

conv_model = create_model()
conv_model.summary()


# Define callbacks, optimizer, loss and metrics

epochs = 50
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='loss', save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_top_k_categorical_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    ),
]
optimizer = tf.keras.optimizers.Adam(amsgrad=True)
loss = tf.keras.losses.CategoricalCrossentropy()


# Compile model and call model.fit()

conv_model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[
        tf.keras.metrics.TopKCategoricalAccuracy(k=3),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ],
)

conv_model_history = conv_model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=test_dataset,
    class_weight=weight_dict,
)


# Visualize model metrics during training

plot_history_metrics(conv_model_history)


# Evaluate model on test data

loss, accuracy, auc, precision, recall = conv_model.evaluate(test_dataset)
print(f'Loss : {loss}')
print(f'Top 3 Categorical Accuracy : {accuracy}')
print(f'Area under the Curve (ROC) : {auc}')
print(f'Precision : {precision}')
print(f'Recall : {recall}')


def view_evaluated_eeg_plots(model):
    start_index = np.random.randint(10, len(eeg))
    end_index = start_index + 11
    data = eeg.loc[start_index:end_index, 'raw_values']
    data_array = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in data]
    data_array = [np.asarray(data_array).astype(np.float32).reshape((-1, 512, 1))]
    original_labels = eeg.loc[start_index:end_index, 'label']
    predicted_labels = np.argmax(model.predict(data_array, verbose=0), axis=1)
    original_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in original_labels
    ]
    predicted_labels = [
        le.inverse_transform(np.array(label).reshape(-1))[0]
        for label in predicted_labels
    ]
    total_plots = 12
    cols = total_plots // 3
    rows = total_plots // cols
    if total_plots % cols:
        rows += 1
    pos = range(1, total_plots + 1)
    fig = plt.figure(figsize=(20, 10))
    for i, (plot_data, og_label, pred_label) in enumerate(
        zip(data, original_labels, predicted_labels)
    ):
        plt.subplot(rows, cols, pos[i])
        plt.plot(plot_data)
        plt.title(f'Actual Label : {og_label}\nPredicted Label : {pred_label}')
        fig.subplots_adjust(hspace=0.5)
    plt.show()


view_evaluated_eeg_plots(conv_model)
