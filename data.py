from tensorflow.keras.preprocessing.text import Tokenizer
import os
import string
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib
import zipfile
import numpy as np
import csv


def get_x_percent_length(x, articles):
    articleLengths = []

    # get a tuple pair of (wordCount, article)
    for article in articles:
        articleLengths.append((len(article.split(" ")), article))

    # sort by article word count
    sorted_lengths = sorted(articleLengths)
    # return length of the article with x percent length
    return sorted_lengths[int((x * len(articles)) / 100)][0]


def process_data_set(DATADIR):

    text = []
    labels = []

    with open(DATADIR) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                text.append(row[0])
                labels.append(int(float(row[1])) - 1)
                line_count += 1

    num_samples = len(text)
    # split 80% into a train set and 20% into a test set
    train = text[0:int(num_samples * .7)]
    train_labels = labels[0:int(num_samples * .7)]
    test = text[int(num_samples * .7): num_samples]
    test_labels = labels[int(num_samples * .7):num_samples]

    # create catagorical array and map labels to catagorical array
    train_cats = tf.keras.utils.to_categorical(train_labels, num_classes=5)
    test_cats = tf.keras.utils.to_categorical(test_labels, num_classes=5)
    return(test, test_cats, train, train_cats)


# I don't know if this is entirely necessary since I am using a predefined tokenizer but I am leaving it in for now
def clean_text(text):

    # define punctuation
    translator = str.maketrans('', '', string.punctuation)
    # remove punctuation and other troublesome characters that appear in the document
    text = text.translate(translator).replace('\n', " ").replace('\t', " ")
    # remove stop words
    STOPWORDS = set(stopwords.words('english'))
    for word in STOPWORDS:
        token = ' ' + word + ' '
        text = text.replace(token, ' ')
        text = text.replace(' ', ' ')
        text = text.lower()

    return text


def createTokenizer(text, vocab_size, oov_token):
    # creates a tokenizer expects an array of strings as text
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_token, lower=True,)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    print(dict(list(word_index.items())[0:10]))
    print(dict(list(word_index.items())[vocab_size-10:vocab_size]))

    return tokenizer


def handle_padding(max_len, text, tokenizer):
    sequences = tokenizer.texts_to_sequences(text)

    print(sequences[10])
    padded = pad_sequences(
        sequences, maxlen=(max_len), padding="post", truncating="post")
    return padded


# Available dimensions for 6B data is 50, 100, 200, 300
def download_glove_dataset(embedding_dimension):
    data_directory = './data/glove'

    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    glove_weights_file_path = os.path.join(
        data_directory, f'glove.6B.{embedding_dimension}d.txt'.format(embedding_dimension))

    if not os.path.isfile(glove_weights_file_path):
        # Glove embedding weights can be downloaded from https://nlp.stanford.edu/projects/glove/
        glove_fallback_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        local_zip_file_path = os.path.join(
            data_directory, os.path.basename(glove_fallback_url))
        if not os.path.isfile(local_zip_file_path):
            print(f'Retreiving glove weights from {glove_fallback_url}')
            urllib.request.urlretrieve(glove_fallback_url, local_zip_file_path)
        with zipfile.ZipFile(local_zip_file_path, 'r') as z:
            print(f'Extracting glove weights from {local_zip_file_path}')
            z.extractall(path=data_directory)


def words_to_glove_idx(text, word2idx, UNKNOWN_TOKEN):

    converted = []
    for article in text:
        current_article = []
        for word in article.split():
            current_article.append(word2idx.get(
                word, UNKNOWN_TOKEN))
        current_article = np.asarray(current_article).astype('float32')
        converted.append(np.array(current_article))
    return converted
