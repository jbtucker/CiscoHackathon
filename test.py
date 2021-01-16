import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import words_to_glove_idx, get_x_percent_length, pad_sequences
from data import process_data_set
from model_with_glove import build_glove_classifier_model, construct_embedding_layer


def convert_to_rating(model_output):
    print(model_output)
    currentMax = 0
    max_idx = 0
    for i in range(len(model_output)):
        if (model_output[i] > currentMax):
            currentMax = model_output[i]
            max_idx = i

    # add one to convert from 0-4 score to 1-5 score
    return max_idx + 1


def rebuild_from_checkpoint(checkpoint_path, example):

    checkpoint_dir = os.path.dirname(checkpoint_path)
    (embedding_layer, word2idx, UNKNOWN_TOKEN) = construct_embedding_layer(400000, 200)
    model = build_glove_classifier_model(400000, 200, embedding_layer)

    model.load_weights(tf.train.latest_checkpoint(
        checkpoint_dir)).expect_partial()

    converted_test = words_to_glove_idx([example], word2idx, UNKNOWN_TOKEN)
    max_len = 329
    padded_test = pad_sequences(
        converted_test, maxlen=(max_len), padding="post", truncating="post")

    example = np.array(padded_test)

    print("rating: ", convert_to_rating(model.predict([example])[0]))


checkpoint_path = "./checkpoints/glove-smaller-lstm/"
negative_example = "I am struggling a lot right now with the material we covered last and I misunderstood the example you showed at the end of class. Right now, I would probably do badly on the quiz. I need you to explain that topic again because it is very unclear still."
positive_example = "Thank you so much Mrs Smith. This has been a great class and has definitely kept me entertained during Covid. Although using remote measures is not ideal I am getting a lot out of this class. I never though calculus could be so much fun."

print("negative example")
rebuild_from_checkpoint(checkpoint_path, negative_example)
print("positive example")
rebuild_from_checkpoint(checkpoint_path, positive_example)
