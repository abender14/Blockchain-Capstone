# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:01:25 2020

@author: alexb
"""

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import h2o

#load dataset for model
dataset = pd.read_csv('h2o_full_data.csv')

#connect to h2o
h2o.connect()

# load the model
h2o_model = h2o.load_model("GBM_1_AutoML_20200120_221705")

#test a result
test_data = [0, 2,18,10,5000,3000,1,12,10,5,6]
test_df = pd.DataFrame(np.array(test_data).reshape(1,11))

#rename columns
x = dataset.columns.tolist()
test_df.columns = x

#convert to h2o frame
df = h2o.H2OFrame(test_df)
preds = h2o_model.predict(df)
preds = preds.as_data_frame()
print(round(100*preds.iloc[0,2],2))