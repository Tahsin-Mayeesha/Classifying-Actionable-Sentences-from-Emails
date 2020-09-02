from fastapi import FastAPI
from rule_model import *
import numpy as np 
import pickle 
import joblib

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional

embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'


with open("models/saved_tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

nb_model = joblib.load("models/nbmodel.joblib")
svm_model = joblib.load("models/svmmodel.joblib")
logreg_model = joblib.load("models/logregmodel.joblib")
lstm_model = tf.keras.models.load_model("models/lstm_model.h5")


app = FastAPI()


@app.post("/")
async def root(message: str,model_name: str="lstm"):
    if model_name == "rule-based":
        prediction = rule_based_model(message)
    elif model_name == "naive bayes":
        prediction = nb_model.predict([message])
    elif model_name == "svm":
        prediction = svm_model.predict([message])
    elif model_name == 'logreg':
        prediction = logreg_model.predict([message])
    else:
        sequences = tokenizer.texts_to_sequences([message])
        pred_padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        prediction = np.argmax(lstm_model.predict(pred_padded), axis=-1)
                         
    return {"message":message, "prediction":str(prediction)}