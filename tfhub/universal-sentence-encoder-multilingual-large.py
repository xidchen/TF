"""
https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3

16 languages (Arabic, Chinese-simplified, Chinese-traditional, English,
French, German, Italian, Japanese, Korean, Dutch, Polish, Portuguese,
Spanish, Thai, Turkish, Russian) text encoder.

Model Details

* Developed by researchers at Google, 2019, v2 [1].
* Transformer.
* Covers 16 languages, showing strong performance on cross-lingual retrieval.
  The input to the module is variable length text in any
  of the aforementioned languages and the output is a 512 dimensional vector.
* Input text can have arbitrary length! However,
  model time and space complexity is $$O(n^2)$$ for input length $$n$$.
  We recommend inputs that are approximately one sentence in length.
* A smaller model with a simpler encoder architecture is available
  that has time and space requirements that scale $$O(n)$$ with input length.

Intended Use

* The model is intended to be used for text classification, text clustering,
  semantic textural similarity retrieval, cross-lingual text retrieval, etc.

Factors

* The Universal Sentence Encoder Multilingual module is an extension
  of the Universal Sentence Encoder Large that includes training
  on multiple tasks across languages.
* The multi-task training setup is based on the paper
  https://arxiv.org/abs/1810.12836.
* This specific module is optimized for multi-word length text,
  such as sentences, phrases or short paragraphs.
* Important: notice that the language of the text input does not need
  to be specified, as the model was trained such that text across languages
  with similar meanings will have close embeddings.

Universal Sentence Encoder family

* There are several versions of universal sentence encoder models trained
  with different goals including size/performance multilingual,
  and fine-grained question answer retrieval.

Prerequisites

* This module relies on the Tensorflow Text for input preprocessing.
"""

import numpy as np
import tensorflow_hub as hub
# noinspection PyUnresolvedReferences
import tensorflow_text


# Some texts of different lengths.
english_sentences = ["dog",
                     "Puppies are nice.",
                     "I enjoy taking long walks along the beach with my dog."]
italian_sentences = ["cane",
                     "I cuccioli sono carini.",
                     "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
japanese_sentences = ["犬",
                      "子犬はいいです",
                      "私は犬と一緒にビーチを散歩するのが好きです"]

embed = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Compute embeddings.
en_result = embed(english_sentences)
it_result = embed(italian_sentences)
ja_result = embed(japanese_sentences)

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_it = np.inner(en_result, it_result)
similarity_matrix_ja = np.inner(en_result, ja_result)
