# dependencies
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# new flask app
app = Flask(__name__)

# load machine learning model
model = pickle.load(open('./notebook/model.pickle', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    features = {
        'passenger_class_1': 0,
        'passenger_class_2': 0,
        'passenger_class_3': 0,
        'gender_female': 0,
        'gender_male': 0,
        'sibling_or_spouse_0': 0,
        'sibling_or_spouse_1': 0,
        'parent_or_child_0': 0,
        'parent_or_child_1': 0,
        'embarked_c': 0,
        'embarked_q': 0,
        'embarked_s': 0,
        'deck_a': 0,
        'deck_b': 0,
        'deck_c': 0,
        'deck_d': 0,
        'deck_e': 0,
        'deck_f': 0,
        'deck_g': 0,
        'deck_t': 0,
        'deck_unavailable': 0,
        'age_categories_adult': 0,
        'age_categories_child': 0,
        'age_categories_middle_age': 0,
        'age_categories_missing': 0,
        'age_categories_senior': 0,
        'age_categories_teenager': 0,
        'age_categories_young_adult': 0,
        'fare_categories_0': 0,
        'fare_categories_1': 0,
        'fare_categories_2': 0,
        'fare_categories_3': 0,
        'fare_categories_4': 0,
        'fare_categories_5': 0,
        'fare_categories_Missing': 0,
        'title_master': 0,
        'title_miss': 0,
        'title_mr': 0,
        'title_mrs': 0,
        'title_other': 0,
    }

    # update features using client data
    jsonData = request.get_json()

    title = jsonData['title']
    features[title] = 1

    gender = jsonData['gender']
    features[gender] = 1

    passenger_class = jsonData['passengerClass']
    features[passenger_class] = 1

    deck = jsonData['cabinLocation']
    features[deck] = 1

    embarkation = jsonData['embarkation']
    features[embarkation] = 1

    # handle age feature
    age = jsonData['age']

    if int(age) < 12:
        age_category = 'age_categories_child'
    elif int(age) < 20:
        age_category = 'age_categories_teenager'
    elif int(age) < 30:
        age_category = 'age_categories_young_adult'
    elif int(age) < 45:
        age_category = 'age_categories_adult'
    elif int(age) < 60:
        age_category = 'age_categories_middle_age'
    else:
        age_category = 'age_categories_senior'

    features[age_category] = 1

    # handle fare feature
    fare = jsonData['fare']

    if int(fare) < 7:
        fare_category = 'fare_categories_0'
    elif int(fare) < 14:
        fare_category = 'fare_categories_1'
    elif int(fare) < 35:
        fare_category = 'fare_categories_2'
    elif int(fare) < 70:
        fare_category = 'fare_categories_3'
    elif int(fare) < 140:
        fare_category = 'fare_categories_4'
    else:
        fare_category = 'fare_categories_5'

    features[fare_category] = 1

    # handle sibling_or_spouse feature
    with_spouse = 1 if jsonData['withSpouse'] else 0
    with_siblings = 1 if jsonData['withSiblings'] else 0

    if with_spouse == 1 or with_siblings == 1:
        sibling_or_spouse_category = 'sibling_or_spouse_1'
    else:
        sibling_or_spouse_category = 'sibling_or_spouse_0'

    features[sibling_or_spouse_category] = 1

    # handle parent_or_child feature
    with_children = 1 if jsonData['withChildren'] else 0
    with_parents = 1 if jsonData['withParents'] else 0

    if with_children == 1 or with_parents == 1:
        parent_or_child_category = 'parent_or_child_1'
    else:
        parent_or_child_category = 'parent_or_child_0'

    features[parent_or_child_category] = 1

    # convert data intto DataFrame
    user_input = pd.DataFrame(np.array([list(features.values())]))

    # make prediction
    prediction = model.predict(user_input)

    return f'{prediction[0]}'
