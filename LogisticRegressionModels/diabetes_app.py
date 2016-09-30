from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

import pickle
import os
import numpy as np

app = FlaskAPI(__name__)
curr_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(curr_dir, 'diabetes_model.sav'), 'rb'))

@app.route("/", methods=['GET'])
def home():
    return {'hello': 'world'}


@app.route("/predict", methods=['POST'])
def predict():
    print(request.data)

    data = [ float(request.data['preg']),
             float(request.data['plas']),
             float(request.data['pres']),
             float(request.data['skin']),
             float(request.data['test']),
             float(request.data['mass']),
             float(request.data['pedi']),
             float(request.data['age'])
            ]

    print(data)
    #m = clf.predict([6, 148, 72, 35, 0, 33.6, 0.627, 50 ])[0]
    m = clf.predict(data)[0]

    print(m)
    return {'prediction': m}

if __name__ == "__main__":
    app.run(debug=True)
