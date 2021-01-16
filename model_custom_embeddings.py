import numpy as np
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub


def build_classifier_model(vocab_size, embedding_dimension):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dimension),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dimension, return_sequences=True)),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embedding_dimension)),

        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(embedding_dimension, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model
