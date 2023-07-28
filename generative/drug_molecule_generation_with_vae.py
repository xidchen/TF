import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem.Draw
import tensorflow as tf

tf.keras.utils.set_random_seed(seed=0)


# Dataset

csv_path = tf.keras.utils.get_file(
    fname="/content/250k_rndm_zinc_drugs_clean_3.csv",
    origin="https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
           "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
)
df = pd.read_csv("/content/250k_rndm_zinc_drugs_clean_3.csv")
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
df.head()
# Downloading data from ...
# 22606589/22606589 [==============================] - 0s 0us/step
# smiles	logP	qed	SAS
# 0	CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1	5.05060	0.702012	2.084095
# 1	C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1	3.11370	0.928975	3.432004
# 2	N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)...	4.96778	0.599682	2.470633
# 3	CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c...	4.00022	0.690944	2.822753
# 4	N#CC1=C(SCC(=O)Nc2cccc(Cl)c2)N=C([O-])[C@H](C#...	3.60956	0.789027	4.035182

# Hyperparameters

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'

bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {
        0: rdkit.Chem.BondType.SINGLE,
        1: rdkit.Chem.BondType.DOUBLE,
        2: rdkit.Chem.BondType.TRIPLE,
        3: rdkit.Chem.BondType.AROMATIC
    }
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

MAX_MOLSIZE = max(df["smiles"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

BATCH_SIZE = 100
EPOCHS = 10

VAE_LR = 5e-4
NUM_ATOMS = 120  # Maximum number of atoms

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types
LATENT_DIM = 435  # Size of the latent space


def smiles_to_graph(smiles):
    # Converts SMILES to molecule object
    molecule = rdkit.Chem.MolFromSmiles(smiles)

    # Initialize adjacency and feature tensor
    _adjacency = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    _features = np.zeros((NUM_ATOMS, ATOM_DIM), "float32")

    # loop over each atom in molecule
    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        _features[i] = np.eye(ATOM_DIM)[atom_type]
        # loop over one-hop neighbors
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            _adjacency[bond_type_idx, [i, j], [j, i]] = 1

    # Where no bond, add 1 to last channel (indicating "non-bond")
    # Notice: channels-first
    _adjacency[-1, np.sum(_adjacency, axis=0) == 0] = 1

    # Where no atom, add 1 to last column (indicating "non-atom")
    _features[np.where(np.sum(_features, axis=1) == 0)[0], -1] = 1

    return _adjacency, _features


def graph_to_molecule(graph):
    # Unpack graph
    _adjacency, _features = graph

    # RWMol is a molecule object intended to be edited
    molecule = rdkit.Chem.RWMol()

    # Remove "no atoms" & atoms with no bonds
    keep_idx = np.where(
        (np.argmax(_features, axis=1) != ATOM_DIM - 1)
        & (np.sum(_adjacency[:-1], axis=(0, 1)) != 0)
    )[0]
    _features = _features[keep_idx]
    _adjacency = _adjacency[:, keep_idx, :][:, :, keep_idx]

    # Add atoms to molecule
    for atom_type_idx in np.argmax(_features, axis=1):
        atom = rdkit.Chem.Atom(atom_mapping[atom_type_idx])
        _ = molecule.AddAtom(atom)

    # Add bonds between atoms in molecule; based on the upper triangles
    # of the [symmetric] adjacency tensor
    (bonds_ij, atoms_i, atoms_j) = np.where(np.triu(_adjacency) == 1)
    for (bond_ij, atom_i, atom_j) in zip(bonds_ij, atoms_i, atoms_j):
        if atom_i == atom_j or bond_ij == BOND_DIM - 1:
            continue
        bond_type = bond_mapping[bond_ij]
        molecule.AddBond(int(atom_i), int(atom_j), bond_type)

    # Sanitize the molecule; for more information on sanitization, see
    # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    flag = rdkit.Chem.SanitizeMol(molecule, catchErrors=True)
    # Let's be strict. If sanitization fails, return None
    if flag != rdkit.Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule


# Generate training set

train_df = df.sample(frac=0.75, random_state=42)  # random state is a seed value
train_df.reset_index(drop=True, inplace=True)

adjacency_tensor, feature_tensor, qed_tensor = [], [], []
for idx in range(8000):
    adjacency, features = smiles_to_graph(train_df.loc[idx]["smiles"])
    qed = train_df.loc[idx]["qed"]
    adjacency_tensor.append(adjacency)
    feature_tensor.append(features)
    qed_tensor.append(qed)

adjacency_tensor = np.array(adjacency_tensor)
feature_tensor = np.array(feature_tensor)
qed_tensor = np.array(qed_tensor)


class RelationalGraphConvLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units=128,
        activation="relu",
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel = None
        self.bias = None
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]
        self.kernel = self.add_weight(
            shape=(bond_dim, atom_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name="W",
            dtype=tf.float32,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(bond_dim, 1, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="b",
                dtype=tf.float32,
            )
        self.built = True

    def call(self, inputs, training=False, **kwargs):
        _adjacency, _features = inputs
        # Aggregate information from neighbors
        x = tf.matmul(_adjacency, _features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.reduce_sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)


# Build the Encoder and Decoder

def get_encoder(
    gconv_units,
    latent_dim,
    adjacency_shape,
    feature_shape,
    dense_units,
    dropout_rate
):
    _adjacency = tf.keras.layers.Input(shape=adjacency_shape)
    _features = tf.keras.layers.Input(shape=feature_shape)

    # Propagate through one or more graph convolutional layers
    features_transformed = _features
    for units in gconv_units:
        features_transformed = RelationalGraphConvLayer(units)(
            [_adjacency, features_transformed]
        )
    # Reduce 2-D representation of molecule to 1-D
    x = tf.keras.layers.GlobalAveragePooling1D()(features_transformed)

    # Propagate through one or more densely connected layers
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    z_mean = tf.keras.layers.Dense(latent_dim, dtype="float32", name="z_mean")(x)
    log_var = tf.keras.layers.Dense(latent_dim, dtype="float32", name="log_var")(x)

    _encoder = tf.keras.Model(
        [_adjacency, _features], [z_mean, log_var], name="encoder"
    )

    return _encoder


def get_decoder(
    dense_units,
    dropout_rate,
    latent_dim,
    adjacency_shape,
    feature_shape
):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))

    x = latent_inputs
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="tanh")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layer (x) to [continuous] adjacency tensors (x_adjacency)
    x_adjacency = tf.keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_adjacency = tf.keras.layers.Reshape(adjacency_shape)(x_adjacency)
    # Symmetrify tensors in the last two dimensions
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
    x_adjacency = tf.keras.layers.Softmax(axis=1)(x_adjacency)

    # Map outputs of previous layer (x) to [continuous] feature tensors (x_features)
    x_features = tf.keras.layers.Dense(tf.math.reduce_prod(feature_shape))(x)
    x_features = tf.keras.layers.Reshape(feature_shape)(x_features)
    x_features = tf.keras.layers.Softmax(axis=2)(x_features)

    _decoder = tf.keras.Model(
        latent_inputs, outputs=[x_adjacency, x_features], name="decoder"
    )

    return _decoder


# Build the Sampling layer

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Build the VAE

class MoleculeGenerator(tf.keras.Model):

    def __init__(self, _encoder, _decoder, max_len, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = None
        self.encoder = _encoder
        self.decoder = _decoder
        self.property_prediction_layer = tf.keras.layers.Dense(1)
        self.max_len = max_len
        self.train_total_loss_tracker = tf.keras.metrics.Mean(name="train_total_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")

    def train_step(self, data):
        _adjacency_tensor, _feature_tensor, _qed_tensor = data[0]
        graph_real = [_adjacency_tensor, _feature_tensor]
        self.batch_size = tf.shape(_qed_tensor)[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, qed_pred, gen_adjacency, gen_features = self(
                graph_real, training=True
            )
            graph_generated = [gen_adjacency, gen_features]
            total_loss = self._compute_loss(
                z_log_var, z_mean, _qed_tensor, qed_pred, graph_real, graph_generated
            )
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.train_total_loss_tracker.update_state(total_loss)
        return {"loss": self.train_total_loss_tracker.result()}

    def _compute_loss(
        self, z_log_var, z_mean, qed_true, qed_pred, graph_real, graph_generated
    ):
        adjacency_real, features_real = graph_real
        adjacency_gen, features_gen = graph_generated
        adjacency_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.categorical_crossentropy(
                    adjacency_real, adjacency_gen
                ),
                axis=(1, 2),
            )
        )
        features_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.categorical_crossentropy(
                    features_real, features_gen
                ),
                axis=1,
            )
        )
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 1
        )
        kl_loss = tf.reduce_mean(kl_loss)
        property_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(qed_true, qed_pred)
        )
        graph_loss = self._gradient_penalty(graph_real, graph_generated)
        return kl_loss + property_loss + graph_loss + adjacency_loss + features_loss

    def _gradient_penalty(self, graph_real, graph_generated):
        # Unpack graphs
        adjacency_real, features_real = graph_real
        adjacency_generated, features_generated = graph_generated
        # Generate interpolated graphs (adjacency_interp and features_interp)
        alpha = tf.random.uniform([self.batch_size])
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = adjacency_real * alpha + (1 - alpha) * adjacency_generated
        alpha = tf.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = features_real * alpha + (1 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            _, _, logits, _, _ = self(
                [adjacency_interp, features_interp], training=True
            )

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        # Compute the gradient penalty
        grads_adjacency_penalty = (1 - tf.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - tf.norm(grads[1], axis=2)) ** 2
        return tf.reduce_mean(
            tf.reduce_mean(grads_adjacency_penalty, axis=(-2, -1))
            + tf.reduce_mean(grads_features_penalty, axis=(-1))
        )

    def inference(self, batch_size):
        z = tf.random.normal((batch_size, LATENT_DIM))
        reconstruction_adjacency, reconstruction_features = self.decoder.predict(z)
        # obtain one-hot encoded adjacency tensor
        _adjacency = tf.argmax(reconstruction_adjacency, axis=1)
        _adjacency = tf.one_hot(_adjacency, depth=BOND_DIM, axis=1)
        # Remove potential self-loops from adjacency
        _adjacency = tf.linalg.set_diag(
            _adjacency, diagonal=tf.zeros(tf.shape(_adjacency)[:-1])
        )
        # obtain one-hot encoded feature tensor
        _features = tf.argmax(reconstruction_features, axis=2)
        _features = tf.one_hot(_features, depth=ATOM_DIM, axis=2)
        return [
            graph_to_molecule([_adjacency[i].numpy(), _features[i].numpy()])
            for i in range(batch_size)
        ]

    def call(self, inputs, **kwargs):
        z_mean, log_var = self.encoder(inputs)
        z = Sampling()([z_mean, log_var])
        gen_adjacency, gen_features = self.decoder(z)
        property_pred = self.property_prediction_layer(z_mean)
        return z_mean, log_var, property_pred, gen_adjacency, gen_features


# Train the model

vae_optimizer = tf.keras.optimizers.Adam(learning_rate=VAE_LR)

encoder = get_encoder(
    gconv_units=[9],
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
    latent_dim=LATENT_DIM,
    dense_units=[512],
    dropout_rate=0.0,
)
decoder = get_decoder(
    dense_units=[128, 256, 512],
    dropout_rate=0.2,
    latent_dim=LATENT_DIM,
    adjacency_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
    feature_shape=(NUM_ATOMS, ATOM_DIM),
)

model = MoleculeGenerator(encoder, decoder, MAX_MOLSIZE)

model.compile(vae_optimizer)
history = model.fit(
    [adjacency_tensor, feature_tensor, qed_tensor],
    epochs=EPOCHS
)
# Epoch 1/10
# 250/250 [==============================] - 8s 12ms/step - loss: 68942.3287
# Epoch 2/10
# 250/250 [==============================] - 3s 12ms/step - loss: 68836.7755
# Epoch 3/10
# 250/250 [==============================] - 3s 11ms/step - loss: 68817.7241
# Epoch 4/10
# 250/250 [==============================] - 3s 11ms/step - loss: 68825.6503
# Epoch 5/10
# 250/250 [==============================] - 3s 11ms/step - loss: 68814.2741
# Epoch 6/10
# 250/250 [==============================] - 3s 12ms/step - loss: 68809.9723
# Epoch 7/10
# 250/250 [==============================] - 3s 12ms/step - loss: 68807.3068
# Epoch 8/10
# 250/250 [==============================] - 3s 12ms/step - loss: 68827.7982
# Epoch 9/10
# 250/250 [==============================] - 3s 11ms/step - loss: 68813.5871
# Epoch 10/10
# 250/250 [==============================] - 3s 12ms/step - loss: 68808.1803

model.summary()
# Model: "molecule_generator_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  encoder (Functional)        [(None, 435),             451925
#                               (None, 435)]
#
#  decoder (Functional)        [(None, 5, 120, 120),     37833576
#                               (None, 120, 11)]
#
#  dense_13 (Dense)            multiple                  436
#
# =================================================================
# Total params: 38,285,941
# Trainable params: 38,285,937
# Non-trainable params: 4
# _________________________________________________________________


# Inference


# Generate unique Molecules with the model

molecules = model.inference(1000)

rdkit.Chem.Draw.MolsToGridImage(
    [m for m in molecules if m is not None][:1000],
    molsPerRow=5,
    subImgSize=(260, 160),
)
# 32/32 [==============================] - 0s 1ms/step
# [11:44:36] non-ring atom 3 marked aromatic
# [11:44:36] Explicit valence for atom # 10 O, 3, is greater than permitted
# [11:44:36] Explicit valence for atom # 5 C, 6, is greater than permitted
# [11:44:36] non-ring atom 2 marked aromatic
# [11:44:36] non-ring atom 2 marked aromatic
# [11:44:36] non-ring atom 2 marked aromatic
# [11:44:37] Explicit valence for atom # 0 S, 153, is greater than permitted
# [11:44:37] Explicit valence for atom # 0 S, 28, is greater than permitted
# ...


# Display latent space clusters with respect to molecular properties (QAE)

def plot_latent(vae, data, labels):
    # display a 2D plot of the property in the latent space
    z_mean, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


plot_latent(
    model, [adjacency_tensor[:8000], feature_tensor[:8000]], qed_tensor[:8000]
)
# 250/250 [==============================] - 1s 5ms/step
