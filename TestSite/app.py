import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

#load in dataset for model training
fullxTrain = pd.read_csv('fullxTrain.csv', nrows=1)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    test_df = pd.DataFrame(np.array(final_features).reshape(1,10))
    x = fullxTrain.columns.tolist()
    test_df.columns = x
    prediction = model.predict_proba(test_df)
    prediction = pd.DataFrame(prediction)
    output = round(100*prediction.iloc[0,0],2)

    return render_template('index.html', prediction_text='The likelihood of the loan being paid back is {}%'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = round(100*prediction.iloc[0,0],2)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
