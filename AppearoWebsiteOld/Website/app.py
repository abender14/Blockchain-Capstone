import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import pandas as pd

#load in dataset for model training
fullxTrain = pd.read_csv('fullxTrain.csv', nrows=1)

app = Flask(__name__)
model = open('model.json', 'rb')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def bitly():
    return render_template('index.html')

@app.route('/model',methods=['GET'])
def ml():
    return render_template('service.html')

@app.route('/blockchain',methods=['GET'])
def block():
    return render_template('blog.html')

@app.route('/company',methods=['GET'])
def company():
    return render_template('contact.html')

@app.route('/demo',methods=['GET'])
def demo():
    return redirect("http://ec2-54-204-237-16.compute-1.amazonaws.com:8080/")

@app.route('/profile',methods=['GET'])
def profile():
    return render_template("portfolio.html")

@app.route('/about',methods=['GET'])
def about():
    return render_template("about.html")

@app.route('/details',methods=['GET'])
def details():
    return render_template("portfolio_details.html")

@app.route('/elements',methods=['GET'])
def elements():
    return render_template("elements.html")

@app.route('/model',methods=['POST'])
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

    return render_template('service.html', prediction_text='The likelihood of the loan being paid back is {}%'.format(output))

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
