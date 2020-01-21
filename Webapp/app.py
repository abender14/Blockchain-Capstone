# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:10:36 2020

@author: alexb
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import h2o 

app = Flask(__name__)

#connect to h2o
h2o.connect()

# load the model
h2o_model = h2o.load_model("GBM_1_AutoML_20200120_221705")

#render webpage
@app.route('/')
def home():
    return render_template('index.html')

#predict from input values with API POST method to machine learning model
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The likelihood of getting paid back is {}%'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)