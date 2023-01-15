import requests
import tensorflow
import urllib.request
from flask import Flask, request, jsonify
from datetime import datetime
import numpy as np
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import string
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer
import json
import nltk
import pandas as pd
# nltk.download('wordnet')
# nltk.download('omw-1.4')


np.random.seed(42)
model = tensorflow.keras.models.load_model("model_fasttext_lstm.h5")

#star Flask application
app = Flask(__name__)


#load tokenizer pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tok = pickle.load(handle)

def to_lower_case(texts):
	texts["title_abstract"] = [entry.lower() for entry in texts["title_abstract"]]
	return texts

def remove_punctuation(texts):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return texts.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def lemmatize_words(texts):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in texts.split()])

def preprocess_text(texts, max_review_length = 120):
    texts = to_lower_case(texts)
    texts = texts["title_abstract"].apply(lambda texts: remove_punctuation(texts))
    texts = texts.apply(lambda texts: lemmatize_words(texts))

    lstm_texts_seq = tok.texts_to_sequences(texts)
    lstm_texts_mat = pad_sequences(lstm_texts_seq, maxlen=max_review_length)
    return lstm_texts_mat

@app.route('/predict',methods=["GET", "POST"])

def predict():
    text = request.args.get('text')
    data_teks = {'title_abstract': text}
    df = (pd.DataFrame(data_teks, index=[0]))
    x = preprocess_text(df)

    predictions = model.predict(x)
    print(predictions)

    if predictions > 0.5:
        value = "include"
    else:
        value = "exclude"
    return value

if __name__ == "__main__":
    # Run locally
    app.run(debug=False)