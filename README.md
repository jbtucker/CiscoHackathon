# Cisco Comp

An attempt to use a Glove based network to preform sentiment analysis on an amazon reviews data set.

You can find the amazon reviews data set used at https://www.kaggle.com/mohitsoni14521452/amazon-product-reviews-data
We filtered by the video games category as the dataset was initially too large.

Contents of Files

## data.py
The data pipeline of the application. Handles the loading, cleaning and tokenizing of the data.

## main.py
The driver of the application. Uses argument parser to allow users to specify settings of the ML model.

## test.py
Allows users to test a trained ml model on example input.

## model_custom_embeddings.py
A tensorflow bidirectional lstm with custom trained word embeddings.

## model_with_bert.py
A tensorflow implementation of BERT embeddings used for sentiment analysis.

## model_with_glove.py
A tensorflow implementation of gloVe embeddings used for sentiment analysis.
  
