# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:10:36 2020

@author: alexb
"""
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import h2o 

#to make future prediction frame
credit_df = pd.read_csv('h2o_full_data.csv',nrows=1)
zero_df = pd.DataFrame(np.array(0).reshape(1,1))
x = credit_df.columns.tolist()

#init app
app = Flask(__name__)

#connect to h2o
h2o.init(ip="127.0.0.1", port ="8080", bind_to_localhost= False)

# load the model
h2o_model = h2o.load_model("GBM_1_AutoML_20200121_172134")

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
    
    final_df = pd.DataFrame(final_features.reshape(1,10))
    final_df = pd.concat([zero_df,final_df], axis=1)
    
    #rename columns
    final_df.columns = x
    
    #convert to h2o frame
    df = h2o.H2OFrame(final_df)
    prediction = h2o_model.predict(df)
    prediction = prediction.as_data_frame()
    
    output = round(100*prediction.iloc[0,2],2)

    return render_template('index.html', prediction_text='The likelihood of not getting paid back is {}%'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    
    data_df = pd.DataFrame(np.array(list(data.values())))
    data_df_h2o = h2o.H2OFrame(data_df)
    prediction = h2o_model.predict(data_df_h2o)

    output = 100*prediction.iloc[0,2]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)