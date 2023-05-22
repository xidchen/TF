import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import typing

tf.keras.utils.set_random_seed(seed=0)


# Hyperparameters

TRAIN_SIZE = 0.5
VAL_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 20
INPUT_SEQUENCE_LENGTH = 12
FORECAST_HORISON = 3
MULTI_HORISON = False
SIGMA2 = 0.1
EPSILON = 0.5
IN_FEATURE = 1
OUT_FEATURE = 10
LSTM_UNITS = 64


# Data preparation

# Data description
# We use a real-world traffic speed dataset named PeMSD7.
# We use the version collected and prepared by Yu et al., 2018.
# The data consists of two files:
#     W_228.csv contains the distances between 228 stations
#     across the District 7 of California.
#     V_228.csv contains traffic speed collected for those stations
#     in the weekdays of May and June 2012.
# The full description of the dataset can be found in Yu et al., 2018.

# Loading data
url = 'https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip'
tf.keras.utils.get_file(origin=url, extract=True)
data_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
distances_array = pd.read_csv(
    os.path.join(data_dir, 'PeMSD7_W_228.csv'), header=None
).to_numpy()
speeds_array = pd.read_csv(
    os.path.join(data_dir, 'PeMSD7_V_228.csv'), header=None
).to_numpy()
print(f'distances_array, shape={distances_array.shape}')
print(f'speeds_array, shape={speeds_array.shape}')

# sub-sampling roads
sample_routes = [
    0,
    1,
    4,
    7,
    8,
    11,
    15,
    108,
    109,
    114,
    115,
    118,
    120,
    123,
    124,
    126,
    127,
    129,
    130,
    132,
    133,
    136,
    139,
    144,
    147,
    216,
]
distances_array = distances_array[np.ix_(sample_routes, sample_routes)]
speeds_array = speeds_array[:, sample_routes]
print(f'distances_array shape={distances_array.shape}')
print(f'speeds_array shape={speeds_array.shape}')

# Splitting and normalizing data


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    """Splits data into train/val/test sets and normalizes the data.
    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent
            the proportion of the dataset to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent
            the proportion of the dataset to include in the validation split.
    Returns:
        `train_array`, `val_array`, `test_array`
    """
    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)
    train_array = (train_array - mean) / std
    val_array = (data_array[num_train:num_train + num_val] - mean) / std
    test_array = (data_array[num_train + num_val:] - mean) / std
    return train_array, val_array, test_array


train_set_array, val_set_array, test_set_array = preprocess(
    speeds_array, TRAIN_SIZE, VAL_SIZE
)
print(f'train set size: {train_set_array.shape}')
print(f'validation set size: {val_set_array.shape}')
print(f'test set size: {test_set_array.shape}')

# Creating TensorFlow Datasets


def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    """Creates tensorflow dataset from numpy array.
    This function creates a dataset
    where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)`
    containing the `input_sequence_length`
    past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.
    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence
            (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values
            of the timeseries for 1 to `forecast_horizon` timesteps ahead.
            If `multi_horizon=False`, the target will be the value
            of the timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        multi_horizon: See `forecast_horizon`.
    Returns:
        A tf.data.Dataset instance.
    """
    inputs = tf.keras.utils.timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        batch_size=batch_size,
        shuffle=False,
    )
    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = tf.keras.utils.timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        batch_size=batch_size,
        shuffle=False,
    )
    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)
    return dataset.prefetch(16).cache()


train_dataset, val_dataset = (
    create_tf_dataset(
        data_array, INPUT_SEQUENCE_LENGTH, FORECAST_HORISON, BATCH_SIZE
    )
    for data_array in [train_set_array, val_set_array]
)
test_dataset = create_tf_dataset(
    test_set_array,
    INPUT_SEQUENCE_LENGTH,
    FORECAST_HORISON,
    batch_size=test_set_array.shape[0],
    shuffle=False,
    multi_horizon=MULTI_HORISON,
)

# Roads Graph


def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.
    It uses the formula
    in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing
    to compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.
    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`.
            Entry `i,j` of this array is the distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied
            to the square `distances` matrix.
        epsilon: A threshold specifying if there is an edge between two nodes.
            Specifically, `A[i,j]=1` if `np.exp(-w2[i,j] / sigma2) >= epsilon`
            and `A[i,j]=0` otherwise, where `A` is the adjacency matrix
            and `w2=distances_array * distances_array`
    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


adjacency_matrix = compute_adjacency_matrix(distances_array, SIGMA2, EPSILON)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f'number of nodes: {graph.num_nodes}, '
      f'number of edges: {len(graph.edges[0])}')


# Network architecture

class GraphConv(tf.keras.layers.Layer):

    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type='mean',
        combination_type='concat',
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=tf.keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype='float32'
            ),
            trainable=True,
        )
        self.activation = tf.keras.layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            'sum': tf.math.unsorted_segment_sum,
            'mean': tf.math.unsorted_segment_mean,
            'max': tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)
        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )
        raise ValueError(f'Invalid aggregation type: {self.aggregation_type}')

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.
        The nodes' representations are obtained
        by multiplying the features tensor with `self.weight`.
        Note that `self.weight` has shape `(in_feat, out_feat)`.
        Args:
            features: Tensor of shape
            `(num_nodes, batch_size, input_seq_len, in_feat)`
        Returns:
            A tensor of shape
            `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(
            features, indices=self.graph_info.edges[1]
        )
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(
        self,
        nodes_representation: tf.Tensor,
        aggregated_messages: tf.Tensor
    ):
        if self.combination_type == 'concat':
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == 'add':
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(
                f'Invalid combination type: {self.combination_type}.'
            )
        return self.activation(h)

    def call(self, features: tf.Tensor, **kwargs):
        """Forward pass.
        Args:
            features: tensor of shape
            `(num_nodes, batch_size, input_seq_len, in_feat)`
        Returns:
            A tensor of shape
            `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


class LSTMGC(tf.keras.layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if graph_conv_params is None:
            graph_conv_params = {
                'aggregation_type': 'mean',
                'combination_type': 'concat',
                'activation': None,
            }
        self.graph_conv = GraphConv(
            in_feat, out_feat, graph_info, **graph_conv_params
        )
        self.lstm = tf.keras.layers.LSTM(lstm_units, activation='relu')
        self.dense = tf.keras.layers.Dense(output_seq_len)
        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs, **kwargs):
        """Forward pass.
        Args:
            inputs: tf.Tensor of shape
            `(batch_size, input_seq_len, num_nodes, in_feat)`
        Returns:
            A tensor of shape
            `(batch_size, output_seq_len, num_nodes)`.
        """
        # convert shape to (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, perm=[2, 0, 1, 3])
        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0], shape[1], shape[2], shape[3],
        )
        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(
            gcn_out, shape=(batch_size * num_nodes, input_seq_len, out_feat)
        )
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)
        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(
            dense_output, shape=(num_nodes, batch_size, self.output_seq_len)
        )
        return tf.transpose(
            output, perm=[1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)


# Model training

st_gcn = LSTMGC(
    IN_FEATURE,
    OUT_FEATURE,
    LSTM_UNITS,
    INPUT_SEQUENCE_LENGTH,
    FORECAST_HORISON,
    graph,
    {
        'aggregation_type': 'mean',
        'combination_type': 'concat',
        'activation': None,
    },
)

model_inputs = tf.keras.layers.Input(
    shape=(INPUT_SEQUENCE_LENGTH, graph.num_nodes, IN_FEATURE)
)
model_outputs = st_gcn(model_inputs)
model = tf.keras.models.Model(model_inputs, model_outputs)
model.compile(
    optimizer=tf.optimizers.RMSprop(learning_rate=0.0002),
    loss=tf.losses.MeanSquaredError(),
)
model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
    validation_data=val_dataset,
)


# Making forecasts on test set

x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)
plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(['actual', 'forecast'])
plt.show()

naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)
print(f'naive MAE: {naive_mse}, model MAE: {model_mse}')
