import pandas as pd
import tensorflow as tf


# Preparing the data

raw_df = pd.read_csv(
    'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
)

val_df = raw_df.sample(frac=0.2, random_state=1337)
train_df = raw_df.drop(val_df.index)
print(f'Using {len(train_df)} samples for training '
      f'and {len(val_df)} samples for validation')


def dataframe_to_dataset(df):
    df = df.copy()
    labels = df.pop('target')
    ds = tf.data.Dataset.from_tensor_slices(tensors=(dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size=32)
    return ds


train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)


# Feature preprocessing with Keras layers

def encode_numerical_feature(feature, name, ds):
    normalizer = tf.keras.layers.Normalization()
    feature_ds = ds.map(lambda x, y: tf.expand_dims(x[name], axis=-1))
    normalizer.adapt(data=feature_ds)
    return normalizer(feature)


def encode_categorical_feature(feature, name, ds, is_string):
    lookup_class = (
        tf.keras.layers.StringLookup
        if is_string else tf.keras.layers.IntegerLookup
    )
    lookup = lookup_class(output_mode='binary')
    feature_ds = ds.map(lambda x, y: tf.expand_dims(x[name], axis=-1))
    lookup.adapt(data=feature_ds)
    return lookup(feature)


# Build a model

sex = tf.keras.Input(shape=(1,), name='sex', dtype='int64')
cp = tf.keras.Input(shape=(1,), name='cp', dtype='int64')
fbs = tf.keras.Input(shape=(1,), name='fbs', dtype='int64')
restecg = tf.keras.Input(shape=(1,), name='restecg', dtype='int64')
exang = tf.keras.Input(shape=(1,), name='exang', dtype='int64')
ca = tf.keras.Input(shape=(1,), name='ca', dtype='int64')

thal = tf.keras.Input(shape=(1,), name='thal', dtype='string')

age = tf.keras.Input(shape=(1,), name='age')
trestbps = tf.keras.Input(shape=(1,), name='trestbps')
chol = tf.keras.Input(shape=(1,), name='chol')
thalach = tf.keras.Input(shape=(1,), name='thalach')
oldpeak = tf.keras.Input(shape=(1,), name='oldpeak')
slope = tf.keras.Input(shape=(1,), name='slope')

all_inputs = [
    sex,
    cp,
    fbs,
    restecg,
    exang,
    ca,
    thal,
    age,
    trestbps,
    chol,
    thalach,
    oldpeak,
    slope,
]

sex_encoded = encode_categorical_feature(sex, 'sex', train_ds, False)
cp_encoded = encode_categorical_feature(cp, 'cp', train_ds, False)
fbs_encoded = encode_categorical_feature(fbs, 'fbs', train_ds, False)
restecg_encoded = encode_categorical_feature(restecg, 'restecg', train_ds, False)
exang_encoded = encode_categorical_feature(exang, 'exang', train_ds, False)
ca_encoded = encode_categorical_feature(ca, 'ca', train_ds, False)

thal_encoded = encode_categorical_feature(thal, 'thal', train_ds, True)

age_encoded = encode_numerical_feature(age, 'age', train_ds)
trestbps_encoded = encode_numerical_feature(trestbps, 'trestbps', train_ds)
chol_encoded = encode_numerical_feature(chol, 'chol', train_ds)
thalach_encoded = encode_numerical_feature(thalach, 'thalach', train_ds)
oldpeak_encoded = encode_numerical_feature(oldpeak, 'oldpeak', train_ds)
slope_encoded = encode_numerical_feature(slope, 'slope', train_ds)

all_features = tf.keras.layers.concatenate(
    [
        sex_encoded,
        cp_encoded,
        fbs_encoded,
        restecg_encoded,
        exang_encoded,
        slope_encoded,
        ca_encoded,
        thal_encoded,
        age_encoded,
        trestbps_encoded,
        chol_encoded,
        thalach_encoded,
        oldpeak_encoded,
    ]
)

hidden_layer = tf.keras.layers.Dense(32, activation='relu')(all_features)
hidden_layer = tf.keras.layers.Dropout(0.5)(hidden_layer)
output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

model = tf.keras.Model(all_inputs, output)
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.losses.BinaryCrossentropy(),
    metrics=[tf.metrics.BinaryAccuracy()]
)

model.fit(train_ds, epochs=20, validation_data=val_ds)


# Inference on new data

sample = {
    'age': 60,
    'sex': 1,
    'cp': 1,
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 2,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 2.3,
    'slope': 3,
    'ca': 0,
    'thal': 'fixed',
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

print(
    f'This particular patient had a {100 * predictions[0][0]:.1f} percent '
    f'probability of having a heart disease, as evaluated by our model.'
)
