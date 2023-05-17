import datasets
import keras_nlp
import tensorflow as tf
import transformers


# Hyperparameters

TRAIN_TEST_SPLIT = 0.1
MAX_INPUT_LENGTH = 1024
MIN_TARGET_LENGTH = 5
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_EPOCHS = 1

MODEL_CHECKPOINT = 't5-small'


# Load datasets

raw_datasets = datasets.load_dataset('xsum', split='train')
raw_datasets = raw_datasets.train_test_split(
    train_size=TRAIN_TEST_SPLIT, test_size=TRAIN_TEST_SPLIT
)


# Data Pre-processing

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

if MODEL_CHECKPOINT in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
    prefix = 'summarize: '
else:
    prefix = ''


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples['document']]
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LENGTH, truncation=True
    )
    labels = tokenizer(
        text_target=examples['summary'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


# Defining the model

model = transformers.TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, return_tensors='tf'
)

train_dataset = tokenized_datasets['train'].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=True,
    collate_fn=data_collator,
)
test_dataset = tokenized_datasets['test'].to_tf_dataset(
    batch_size=BATCH_SIZE,
    columns=['input_ids', 'attention_mask', 'labels'],
    shuffle=False,
    collate_fn=data_collator,
)
generation_dataset = (
    tokenized_datasets['test']
    .shuffle()
    .select(list(range(200)))
    .to_tf_dataset(
        batch_size=BATCH_SIZE,
        columns=['input_ids', 'attention_mask', 'labels'],
        shuffle=False,
        collate_fn=data_collator,
    )
)


# Building and Compiling the model

model.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE))


# Training and Evaluating the model

rouge_l = keras_nlp.metrics.RougeL()


def metric_fn(eval_predictions):
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_l(decoded_labels, decoded_predictions)
    result = {'RougeL': result['f1_score']}
    return result


metric_callback = transformers.KerasMetricCallback(
    metric_fn=metric_fn,
    eval_dataset=generation_dataset,
    label_cols=['labels'],
    predict_with_generate=True,
)

model.fit(
    train_dataset,
    epochs=MAX_EPOCHS,
    callbacks=[metric_callback],
    validation_data=test_dataset,
)


# Inference

summarizer = transformers.pipeline(
    task='summarization', model=model, tokenizer=tokenizer, framework='tf'
)

summarizer(
    raw_datasets['test'][0]['document'],
    min_length=MIN_TARGET_LENGTH,
    max_length=MAX_TARGET_LENGTH,
)
