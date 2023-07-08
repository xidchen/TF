import keras_nlp
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import time

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)


# General hyperparameters
BATCH_SIZE = 32
NUM_BATCHES = 500
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 200

GPT2_PRESET = 'gpt2_base_en'


# LoRA-specific hyperparameters
RANK = 4
ALPHA = 32


# DATASET

reddit_ds = tfds.load('reddit_tifu', split='train', as_supervised=True)

train_ds = (
    reddit_ds
    .map(lambda document, _: document)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
    .take(NUM_BATCHES)
)


# Helper functions

# Callback for tracking GPU memory usage
class GPUMemoryCallback(tf.keras.callbacks.Callback):

    def __init__(self,
                 target_batches,
                 print_stats=False):
        super().__init__()
        self.target_batches = target_batches
        self.print_stats = print_stats
        self.memory_usage = []
        self.labels = []

    def _compute_memory_usage(self):
        memory_stats = tf.config.experimental.get_memory_info('GPU:0')
        peak_usage = round(memory_stats['peak'] / 2 ** 30, 3)
        self.memory_usage.append(peak_usage)

    def on_epoch_begin(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f'epoch {epoch} start')

    def on_train_batch_begin(self, batch, logs=None):
        if batch in self.target_batches:
            self._compute_memory_usage()
            self.labels.append(f'batch {batch}')

    def on_epoch_end(self, epoch, logs=None):
        self._compute_memory_usage()
        self.labels.append(f'epoch {epoch} end')


# Function for text generation
def generate_text(model, input_text, max_length=200):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    print(f'\nOutput: {output}')
    end = time.time()
    print(f'Total Time Elapsed: {end - start:.2f}s')


# Define optimizer and loss
def get_optimizer_and_loss():
    _optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5,
        epsilon=1e-6,
        weight_decay=0.01,
        global_clipnorm=1.0,
    )
    _optimizer.exclude_from_weight_decay(var_names=['bias'])
    _optimizer.exclude_from_weight_decay(var_names=['beta'])
    _optimizer.exclude_from_weight_decay(var_names=['gamma'])
    _loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return _optimizer, _loss


# Fine-tune GPT-2

preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    'gpt2_base_en',
    sequence_length=MAX_SEQUENCE_LENGTH,
)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
    'gpt2_base_en',
    preprocessor=preprocessor,
)
gpt2_lm.summary()

gpu_memory_callback = GPUMemoryCallback(
    target_batches=[5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
    print_stats=True,
)
optimizer, loss = get_optimizer_and_loss()
gpt2_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=['accuracy'],
)

gpt2_lm.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[gpu_memory_callback],
)
gpt2_lm_memory_usage = gpu_memory_callback.memory_usage

generate_text(
    gpt2_lm,
    'I like basketball',
    max_length=MAX_GENERATION_LENGTH,
)
generate_text(
    gpt2_lm,
    'That Italian restaurant is',
    max_length=MAX_GENERATION_LENGTH,
)


# LoRA GPT-2

# Create LoRA layer
class LoraLayer(tf.keras.layers.Layer):

    def __init__(self,
                 original_layer,
                 rank=8,
                 alpha=32,
                 trainable=False,
                 **kwargs):
        original_layer_config = original_layer.get_config()
        name = original_layer_config['name']
        kwargs.pop('name', None)
        super().__init__(name=name, trainable=trainable, **kwargs)
        self.rank = rank
        self.alpha = alpha
        self._scale = alpha / rank
        self._num_heads = original_layer_config['output_shape'][-2]
        self._hidden_dim = self._num_heads * original_layer_config['output_shape'][-1]
        self.original_layer = original_layer
        self.original_layer.trainable = False
        self.A = tf.keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode='fan_in', distribution='uniform'
            ),
            trainable=trainable,
            name=f'lora_A',
        )
        self.B = tf.keras.layers.EinsumDense(
            equation=original_layer_config['equation'],
            output_shape=original_layer_config['output_shape'],
            kernel_initializer='zeros',
            trainable=trainable,
            name=f'lora_B'
        )

    def call(self, inputs, *args, **kwargs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output
        return original_output


# Inject LoRA layer into the model
del gpt2_lm
del optimizer
del loss

tf.config.experimental.reset_memory_stats('GPU:0')
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
    'gpt2_base_en',
    sequence_length=MAX_SEQUENCE_LENGTH,
)
lora_model = keras_nlp.models.GPT2CausalLM.from_preset(
    'gpt2_base_en',
    preprocessor=preprocessor,
)

for layer_idx in range(lora_model.backbone.num_layers):
    decoder_layer = lora_model.backbone.get_layer(f'transformer_layer_{layer_idx}')
    self_attention_layer = decoder_layer._self_attention_layer
    self_attention_layer._query_dense = LoraLayer(
        self_attention_layer._query_dense,
        rank=RANK,
        alpha=ALPHA,
        trainable=True,
    )
    self_attention_layer._value_dense = LoraLayer(
        self_attention_layer._value_dense,
        rank=RANK,
        alpha=ALPHA,
        trainable=True,
    )

lora_model(preprocessor(['LoRA is very useful for quick LLM finetuning'])[0])
pass

for layer in lora_model._flatten_layers():
    list_of_sublayers = list(layer._flatten_layers())
    if len(list_of_sublayers) == 1:
        if layer.name in ['lora_A', 'lora_B']:
            layer.trainable = True
        else:
            layer.trainable = False

lora_model.summary()


# Fine-tune LoRA GPT-2
gpu_memory_callback = GPUMemoryCallback(
    target_batches=[5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
    print_stats=True,
)
optimizer, loss = get_optimizer_and_loss()
lora_model.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=['accuracy'],
)
lora_model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[gpu_memory_callback],
)
lora_model_memory_usage = gpu_memory_callback.memory_usage

plt.bar(
    x=['GPT-2', 'LoRA GPT-2'],
    height=[max(gpt2_lm_memory_usage), max(lora_model_memory_usage)],
    color=['red', 'blue'],
)
plt.xlabel('Time')
plt.ylabel('GPU Memory Usage (in GB)')
plt.title('GPU Memory Usage Comparison')
plt.legend()
plt.show()


# Merge weights and generate text!
for layer_idx in range(lora_model.backbone.num_layers):
    decoder_layer = lora_model.backbone.get_layer(f'transformer_layer_{layer_idx}')
    self_attention_layer = decoder_layer._self_attention_layer

    query_lora_layer = self_attention_layer._query_dense
    A_weights = query_lora_layer.A.kernel
    B_weights = query_lora_layer.B.kernel
    increment_weights = tf.einsum('ab,bcd->acd', A_weights, B_weights) * ALPHA / RANK
    query_lora_layer.original_layer.kernel.assign_add(increment_weights)

    value_lora_layer = self_attention_layer._value_dense
    A_weights = value_lora_layer.A.kernel
    B_weights = value_lora_layer.B.kernel
    increment_weights = tf.einsum('ab,bcd->acd', A_weights, B_weights) * ALPHA / RANK
    value_lora_layer.original_layer.kernel.assign_add(increment_weights)

generate_text(
    lora_model,
    'I like basketball',
    max_length=MAX_GENERATION_LENGTH,
)
generate_text(
    lora_model,
    'That Italian restaurant is',
    max_length=MAX_GENERATION_LENGTH,
)
