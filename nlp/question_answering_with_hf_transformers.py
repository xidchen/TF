import datasets
import tensorflow as tf
import transformers


# Loading the dataset

ds = datasets.load_dataset('squad')


# Preprocessing the training data

model_checkpoint = 'distilbert-base-cased'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384  # The maximum length of a feature (question and context)
doc_stride = (
    128  # The authorized overlap between two part of the context when splitting
)
# it is needed.


def prepare_train_features(examples):
    examples['question'] = [q.lstrip() for q in examples['question']]
    examples['context'] = [c.lstrip() for c in examples['context']]
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples.pop('offset_mapping')
    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        if len(answers['answer_start']) == 0:
            tokenized_examples['start_positions'].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples['start_positions'].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples['end_positions'].append(token_end_index + 1)
    return tokenized_examples


tokenized_datasets = ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=ds['train'].column_names,
    num_proc=3,
)

train_set = tokenized_datasets['train'].with_format('numpy')[:]
validation_set = tokenized_datasets['validation'].with_format('numpy')[:]


# Fine-tuning the model

model = transformers.TFAutoModelForQuestionAnswering.from_pretrained(
    model_checkpoint
)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam(learning_rate=5e-5)
)

model.fit(train_set, epochs=3, validation_data=validation_set)


# Inference

context = """Keras is an API designed for human beings, not machines. Keras follows best
practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes
the number of user actions required for common use cases, and it provides clear &
actionable error messages. It also has extensive documentation and developer guides. """
question = 'What is Keras?'

inputs = tokenizer([context], [question], return_tensors='np')
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))

answer = inputs['input_ids'][0, int(start_position):int(end_position) + 1]
print(answer)
print(tokenizer.decode(answer))

model.push_to_hub('transformers-qa', organization='keras-io')
tokenizer.push_to_hub('transformers-qa', organization='keras-io')
