import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf

tf.keras.utils.set_random_seed(seed=0)


# Prepare the data

# Download and prepare the DataFrames
ROOT_DIR = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')
zip_data_path = tf.keras.utils.get_file(
    origin='https://files.grouplens.org/datasets/movielens/ml-1m.zip',
    extract=True,
)

users = pd.read_csv(
    os.path.join(DATA_DIR, 'users.dat'),
    sep='::',
    names=['user_id', 'sex', 'age_group', 'occupation', 'zip_code'],
    engine='python',
)
ratings = pd.read_csv(
    os.path.join(DATA_DIR, 'ratings.dat'),
    sep='::',
    names=['user_id', 'movie_id', 'rating', 'unix_timestamp'],
    engine='python',
)
movies = pd.read_csv(
    os.path.join(DATA_DIR, 'movies.dat'),
    sep='::',
    names=['movie_id', 'title', 'genres'],
    engine='python',
    encoding='iso-8859-1',
)

users['user_id'] = users['user_id'].apply(lambda x: f'user_{x}')
users['age_group'] = users['age_group'].apply(lambda x: f'group_{x}')
users['occupation'] = users['occupation'].apply(lambda x: f'occupation_{x}')

movies['movie_id'] = movies['movie_id'].apply(lambda x: f'movie_{x}')

ratings['movie_id'] = ratings['movie_id'].apply(lambda x: f'movie_{x}')
ratings['user_id'] = ratings['user_id'].apply(lambda x: f'user_{x}')
ratings['rating'] = ratings['rating'].apply(lambda x: float(x))

genres = [
    'Action',
    'Adventure',
    'Animation',
    'Children\'s',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western',
]
for genre in genres:
    movies[genre] = movies['genres'].apply(
        lambda values: int(genre in values.split('|'))
    )

# Transform the movie ratings data into sequences
ratings_group = ratings.sort_values(by=['unix_timestamp']).groupby('user_id')
ratings_data = pd.DataFrame(
    data={
        'user_id': list(ratings_group.groups.keys()),
        'movie_ids': list(ratings_group.movie_id.apply(list)),
        'ratings': list(ratings_group.rating.apply(list)),
        'timestamps': list(ratings_group.unix_timestamp.apply(list)),
    }
)

SEQ_LENGTH = 4
SEQ_STEP_SIZE = 2


def create_sequences(values, window_size, step_size):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences


ratings_data.movie_ids = ratings_data.movie_ids.apply(
    lambda ids: create_sequences(ids, SEQ_LENGTH, SEQ_STEP_SIZE)
)
ratings_data.ratings = ratings_data.ratings.apply(
    lambda ids: create_sequences(ids, SEQ_LENGTH, SEQ_STEP_SIZE)
)
del ratings_data['timestamps']

ratings_data_movies = ratings_data[['user_id', 'movie_ids']].explode(
    'movie_ids', ignore_index=True
)
ratings_data_rating = ratings_data[['ratings']].explode(
    'ratings', ignore_index=True)
ratings_data_transformed = pd.concat(
    [ratings_data_movies, ratings_data_rating], axis=1
)
ratings_data_transformed = ratings_data_transformed.join(
    users.set_index('user_id'), on='user_id'
)
ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(
    lambda x: ','.join(x)
)
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
    lambda x: ','.join([str(v) for v in x])
)

del ratings_data_transformed['zip_code']

ratings_data_transformed.rename(
    columns={'movie_ids': 'sequence_movie_ids', 'ratings': 'sequence_ratings'},
    inplace=True,
)

random_selection = np.random.rand(len(ratings_data_transformed.index)) <= 0.85
train_data = ratings_data_transformed[random_selection]
test_data = ratings_data_transformed[~random_selection]

train_data.to_csv('train_data.csv', index=False, sep='|', header=False)
test_data.to_csv('test_data.csv', index=False, sep='|', header=False)


# Define metadata

CSV_HEADER = list(ratings_data_transformed.columns)
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    'user_id': list(users.user_id.unique()),
    'movie_id': list(movies.movie_id.unique()),
    'sex': list(users.sex.unique()),
    'age_group': list(users.age_group.unique()),
    'occupation': list(users.occupation.unique()),
}
USER_FEATURES = ['sex', 'age_group', 'occupation']
MOVIE_FEATURES = ['genres']


# Create tf.data.Dataset for training and evaluation

def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):

    def process(features):
        movie_ids_string = features['sequence_movie_ids']
        sequence_movie_ids = tf.strings.split(movie_ids_string, ',').to_tensor()
        # The last movie id in the sequence is the target movie.
        features['target_movie_id'] = sequence_movie_ids[:, -1]
        features['sequence_movie_ids'] = sequence_movie_ids[:, :-1]
        ratings_string = features['sequence_ratings']
        sequence_ratings = tf.strings.to_number(
            tf.strings.split(ratings_string, ','), tf.dtypes.float32
        ).to_tensor()
        # The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features['sequence_ratings'] = sequence_ratings[:, :-1]
        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        num_epochs=1,
        header=False,
        field_delim='|',
        shuffle=shuffle,
    ).map(process)
    return dataset


# Create model inputs

def create_model_inputs():
    return {
        'user_id': tf.keras.Input(
            name='user_id', shape=(1,), dtype=tf.string
        ),
        'sequence_movie_ids': tf.keras.Input(
            name='sequence_movie_ids', shape=(SEQ_LENGTH - 1,), dtype=tf.string
        ),
        'target_movie_id': tf.keras.Input(
            name='target_movie_id', shape=(1,), dtype=tf.string
        ),
        'sequence_ratings': tf.keras.Input(
            name='sequence_ratings', shape=(SEQ_LENGTH - 1,), dtype=tf.float32
        ),
        'sex': tf.keras.Input(
            name='sex', shape=(1,), dtype=tf.string
        ),
        'age_group': tf.keras.Input(
            name='age_group', shape=(1,), dtype=tf.string
        ),
        'occupation': tf.keras.Input(
            name='occupation', shape=(1,), dtype=tf.string
        ),
    }


# Encode input features

def encode_input_features(
    inputs,
    include_user_id=True,
    include_user_features=True,
    include_movie_features=True,
):
    encoded_transformer_features = []
    encoded_other_features = []
    other_feature_names = []
    if include_user_id:
        other_feature_names.append('user_id')
    if include_user_features:
        other_feature_names.extend(USER_FEATURES)
    # Encode user features
    for feature_name in other_feature_names:
        # Convert the string input values into integer indices
        vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
        idx = tf.keras.layers.StringLookup(
            vocabulary=vocabulary, mask_token=None, num_oov_indices=0
        )(inputs[feature_name])
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = tf.keras.layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f'{feature_name}_embedding',
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))
    # Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = tf.keras.layers.concatenate(
            encoded_other_features
        )
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None
    # Create a movie embedding encoder
    movie_vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY['movie_id']
    movie_embedding_dims = int(math.sqrt(len(movie_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    movie_index_lookup = tf.keras.layers.StringLookup(
        vocabulary=movie_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name='movie_index_lookup',
    )
    # Create an embedding layer with the specified dimensions.
    movie_embedding_encoder = tf.keras.layers.Embedding(
        input_dim=len(movie_vocabulary),
        output_dim=movie_embedding_dims,
        name=f'movie_embedding',
    )
    # Create a vector lookup for movie genres.
    genre_vectors = movies[genres].to_numpy()
    movie_genres_lookup = tf.keras.layers.Embedding(
        input_dim=genre_vectors.shape[0],
        output_dim=genre_vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(genre_vectors),
        trainable=False,
        name='genres_vector',
    )
    # Create a processing layer for genres.
    movie_embedding_processor = tf.keras.layers.Dense(
        units=movie_embedding_dims,
        activation='relu',
        name='process_movie_embedding_with_genres',
    )

    # Define a function to encode a given movie id.
    def encode_movie(movie_id):
        # Convert the string input values into integer indices.
        movie_idx = movie_index_lookup(movie_id)
        movie_embedding = movie_embedding_encoder(movie_idx)
        encoded_movie_embedding = movie_embedding
        if include_movie_features:
            movie_genres_vector = movie_genres_lookup(movie_idx)
            encoded_movie_embedding = movie_embedding_processor(
                tf.keras.layers.concatenate(
                    [movie_embedding, movie_genres_vector]
                )
            )
        return encoded_movie_embedding

    # Encoding target_movie_id
    target_movie_id = inputs['target_movie_id']
    encoded_target_movie = encode_movie(target_movie_id)
    # Encoding sequence movie_ids.
    sequence_movies_ids = inputs['sequence_movie_ids']
    encoded_sequence_movies = encode_movie(sequence_movies_ids)
    # Create positional embedding.
    position_embedding_encoder = tf.keras.layers.Embedding(
        input_dim=SEQ_LENGTH,
        output_dim=movie_embedding_dims,
        name='position_embedding',
    )
    positions = tf.range(start=0, limit=SEQ_LENGTH - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into encoding of the movie.
    sequence_ratings = tf.expand_dims(inputs['sequence_ratings'], -1)
    # Add positional encoding to movie encodings and multiply them by rating.
    encoded_sequence_movies_with_position_and_rating = (
        tf.keras.layers.Multiply()(
            [(encoded_sequence_movies + encodded_positions), sequence_ratings]
        )
    )
    # Construct the transformer inputs.
    for encoded_movie in tf.unstack(
        encoded_sequence_movies_with_position_and_rating, axis=1
    ):
        encoded_transformer_features.append(tf.expand_dims(encoded_movie, 1))
    encoded_transformer_features.append(encoded_target_movie)
    encoded_transformer_features = tf.keras.layers.concatenate(
        encoded_transformer_features, axis=1
    )
    return encoded_transformer_features, encoded_other_features


# Create a BST model

hidden_units = [256, 128]
dropout_rate = 0.1
num_heads = 3


def create_model():
    inputs = create_model_inputs()
    transformer_features, other_features = encode_input_features(
        inputs, 
        include_user_id=False, 
        include_user_features=False, 
        include_movie_features=False
    )

    # Create a multi-headed attention layer.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=transformer_features.shape[2], 
        dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    x1 = tf.keras.layers.Add()([transformer_features, attention_output])
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x2 = tf.keras.layers.LeakyReLU()(x1)
    x2 = tf.keras.layers.Dense(units=x2.shape[-1])(x2)
    x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
    transformer_features = tf.keras.layers.Add()([x1, x2])
    transformer_features = tf.keras.layers.LayerNormalization()(
        transformer_features
    )
    features = tf.keras.layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = tf.keras.layers.concatenate([
            features,
            tf.keras.layers.Reshape([other_features.shape[-1]])(other_features)
        ])

    # Fully-connected layers.
    for num_units in hidden_units:
        features = tf.keras.layers.Dense(num_units)(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.LeakyReLU()(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)

    outputs = tf.keras.layers.Dense(units=1)(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


model = create_model()


# Run training and evaluation experiment

model.compile(
    optimizer=tf.optimizers.Adagrad(learning_rate=0.01),
    loss=tf.losses.MeanSquaredError(),
    metrics=[tf.metrics.MeanAbsoluteError()],
)

train_dataset = get_dataset_from_csv(
    'train_data.csv', shuffle=True, batch_size=265
)

model.fit(train_dataset, epochs=5)

test_dataset = get_dataset_from_csv('test_data.csv', batch_size=265)

_, rmse = model.evaluate(test_dataset, verbose=0)
print(f'Test MAE: {round(rmse, 3)}')
