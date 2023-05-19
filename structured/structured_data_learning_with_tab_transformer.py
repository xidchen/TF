import pandas as pd
import tensorflow as tf

tf.keras.utils.set_random_seed(seed=0)


# Prepare the data

CSV_HEADER = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'income_bracket',
]

train_data_url = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
)
train_data = pd.read_csv(train_data_url, header=None, names=CSV_HEADER)

test_data_url = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
)
test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)

print(f'Train dataset shape: {train_data.shape}')
print(f'Test dataset shape: {test_data.shape}')

test_data = test_data[1:]
test_data.income_bracket = test_data.income_bracket.apply(
    lambda value: value.replace('.', '')
)

train_data_file = 'train_data.csv'
test_data_file = 'test_data.csv'

train_data.to_csv(train_data_file, index=False, header=False)
test_data.to_csv(test_data_file, index=False, header=False)


# Define dataset metadata

NUMERIC_FEATURE_NAMES = [
    'age',
    'education_num',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
]

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    'workclass': sorted(list(train_data['workclass'].unique())),
    'education': sorted(list(train_data['education'].unique())),
    'marital_status': sorted(list(train_data['marital_status'].unique())),
    'occupation': sorted(list(train_data['occupation'].unique())),
    'relationship': sorted(list(train_data['relationship'].unique())),
    'race': sorted(list(train_data['race'].unique())),
    'gender': sorted(list(train_data['gender'].unique())),
    'native_country': sorted(list(train_data['native_country'].unique())),
}

WEIGHT_COLUMN_NAME = 'fnlwgt'

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

COLUMN_DEFAULTS = [
    [0.0]
    if feature_name in NUMERIC_FEATURE_NAMES + [WEIGHT_COLUMN_NAME] else ['NA']
    for feature_name in CSV_HEADER
]

TARGET_FEATURE_NAME = 'income_bracket'

TARGET_LABELS = [' <=50K', ' >50K']


# Configure the hyperparameters

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2
BATCH_SIZE = 265
NUM_EPOCHS = 15

NUM_TRANSFORMER_BLOCKS = 3
NUM_HEADS = 4
EMBEDDING_DIMS = 16
MLP_HIDDEN_UNITS_FACTORS = [
    2,
    1,
]  # MLP hidden layer units, as factors of the number of inputs.
NUM_MLP_BLOCKS = 2  # Number of MLP blocks in the baseline model.


# Implement data reading pipeline

target_label_lookup = tf.keras.layers.StringLookup(
    mask_token=None, num_oov_indices=0, vocabulary=TARGET_LABELS,
)


def prepare_example(features, target):
    target_index = target_label_lookup(target)
    weights = features.pop(WEIGHT_COLUMN_NAME)
    return features, target_index, weights


def get_dataset_from_csv(csv_file_path, batch_size=128, shuffle=False):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        na_value='?',
        shuffle=shuffle,
    ).map(
        prepare_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return dataset.cache()


# Implement a training and evaluation procedure

def run_experiment(
    model,
    train_file,
    test_file,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    optimizer = tf.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.BinaryCrossentropy(),
        metrics=[tf.metrics.BinaryAccuracy(name='accuracy')],
        weighted_metrics=[],
    )
    train_dataset = get_dataset_from_csv(train_file, batch_size, shuffle=True)
    validation_dataset = get_dataset_from_csv(test_file, batch_size)
    print('Start training the model...')
    history = model.fit(
        train_dataset, epochs=num_epochs, validation_data=validation_dataset
    )
    print('Model training finished')
    _, accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f'Validation accuracy: {round(accuracy * 100, 2)}%')
    return history


# Create model inputs

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = tf.keras.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = tf.keras.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


# Encode features

def encode_inputs(inputs, embedding_dims):
    encoded_categorical_feature_list = []
    numerical_feature_list = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any oov token,
            # we set mask_token to None and num_oov_indices to 0.
            lookup = tf.keras.layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode='int',
            )
            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])
            # Create an embedding layer with the specified dimensions.
            embedding = tf.keras.layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )
            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)
        else:
            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], axis=-1)
            numerical_feature_list.append(numerical_feature)
    return encoded_categorical_feature_list, numerical_feature_list


# Implement an MLP block

def create_mlp(
    hidden_units, dropout_rate, activation, normalization_layer, name=None
):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))
    return tf.keras.Sequential(mlp_layers, name=name)


# Experiment 1: a baseline model

def create_baseline_model(
    embedding_dims, num_mlp_blocks, mlp_hidden_units_factors, dropout_rate
):
    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Concatenate all features.
    features = tf.keras.layers.concatenate(
        encoded_categorical_feature_list + numerical_feature_list
    )
    # Compute Feedforward layer units.
    feedforward_units = [features.shape[-1]]
    # Create several feedforwad layers with skip connections.
    for layer_idx in range(num_mlp_blocks):
        features = create_mlp(
            hidden_units=feedforward_units,
            dropout_rate=dropout_rate,
            activation=tf.keras.activations.gelu,
            normalization_layer=tf.keras.layers.LayerNormalization(epsilon=1e-6),
            name=f'feedforward_{layer_idx}',
        )(features)
    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=tf.keras.activations.selu,
        normalization_layer=tf.keras.layers.BatchNormalization(),
        name='MLP',
    )(features)
    # Add a sigmoid as a binary classifer.
    outputs = tf.keras.layers.Dense(
        units=1, activation='sigmoid', name='sigmoid'
    )(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


baseline_model = create_baseline_model(
    embedding_dims=EMBEDDING_DIMS,
    num_mlp_blocks=NUM_MLP_BLOCKS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)
print(f'Total model weights: {baseline_model.count_params()}')

baseline_model_history = run_experiment(
    model=baseline_model,
    train_file=train_data_file,
    test_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)


# Experiment 2: TabTransformer

def create_tabtransformer_classifier(
    num_transformer_blocks,
    num_heads,
    embedding_dims,
    mlp_hidden_units_factors,
    dropout_rate,
    use_column_embedding=False,
):
    # Create model inputs.
    inputs = create_model_inputs()
    # encode features.
    encoded_categorical_feature_list, numerical_feature_list = encode_inputs(
        inputs, embedding_dims
    )
    # Stack categorical feature embeddings for the Transformer.
    encoded_categorical_features = tf.stack(
        encoded_categorical_feature_list, axis=1
    )
    # Concatenate numerical features.
    numerical_features = tf.keras.layers.concatenate(numerical_feature_list)
    # Add column embedding to categorical feature embeddings.
    if use_column_embedding:
        num_columns = encoded_categorical_features.shape[1]
        column_embedding = tf.keras.layers.Embedding(
            input_dim=num_columns, output_dim=embedding_dims
        )
        column_indices = tf.range(start=0, limit=num_columns, delta=1)
        encoded_categorical_features = (
            encoded_categorical_features + column_embedding(column_indices)
        )
    # Create multiple layers of the Transformer block.
    for block_idx in range(num_transformer_blocks):
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dims,
            dropout=dropout_rate,
            name=f'multihead_attention_{block_idx}',
        )(encoded_categorical_features, encoded_categorical_features)
        # Skip connection 1.
        x = tf.keras.layers.Add(name=f'skip_connection1_{block_idx}')(
            [attention_output, encoded_categorical_features]
        )
        # Layer normalization 1.
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'layer_norm1_{block_idx}'
        )(x)
        # Feedforward.
        feedforward_output = create_mlp(
            hidden_units=[embedding_dims],
            dropout_rate=dropout_rate,
            activation=tf.keras.activations.gelu,
            normalization_layer=tf.keras.layers.LayerNormalization(epsilon=1e-6),
            name=f'feedforward_{block_idx}',
        )(x)
        # Skip connection 2.
        x = tf.keras.layers.Add(name=f'skip_connection2_{block_idx}')(
            [feedforward_output, x]
        )
        # Layer normalization 2.
        encoded_categorical_features = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'layer_norm2_{block_idx}'
        )(x)
    # Flatten the 'contextualized' embeddings of the categorical features.
    categorical_features = tf.keras.layers.Flatten()(
        encoded_categorical_features
    )
    # Apply layer normalization to the numerical features.
    numerical_features = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
        numerical_features
    )
    # Prepare the input for the final MLP block.
    features = tf.keras.layers.concatenate(
        [categorical_features, numerical_features]
    )
    # Compute MLP hidden_units.
    mlp_hidden_units = [
        factor * features.shape[-1] for factor in mlp_hidden_units_factors
    ]
    # Create final MLP.
    features = create_mlp(
        hidden_units=mlp_hidden_units,
        dropout_rate=dropout_rate,
        activation=tf.keras.activations.selu,
        normalization_layer=tf.keras.layers.BatchNormalization(),
        name='MLP',
    )(features)
    # Add a sigmoid as a binary classifer.
    outputs = tf.keras.layers.Dense(
        units=1, activation='sigmoid', name='sigmoid'
    )(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


tabtransformer_model = create_tabtransformer_classifier(
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    embedding_dims=EMBEDDING_DIMS,
    mlp_hidden_units_factors=MLP_HIDDEN_UNITS_FACTORS,
    dropout_rate=DROPOUT_RATE,
)

print(f'Total model weights: {tabtransformer_model.count_params()}')

tabtransformer_model_history = run_experiment(
    model=tabtransformer_model,
    train_file=train_data_file,
    test_file=test_data_file,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)
