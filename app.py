# dependencies
from flask import Flask, request
import numpy as np
import pandas as pd
import pickle

# new flask app and configure static directory
app = Flask(__name__, static_folder='./client/build', static_url_path='/')

# load machine learning model
model = pickle.load(open('./notebook/model.pickle', 'rb'))


# routes
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    return request.form
