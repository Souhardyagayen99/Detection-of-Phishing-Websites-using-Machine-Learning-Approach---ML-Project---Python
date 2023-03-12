from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import false, true
import feature

app = Flask(__name__)
model = pickle.load(open('SVM.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',isGood="")


@app.route('/predict',methods=['POST'])
def predict():
    output = request.form.get("urlname")
    prediction = model.predict([feature.generate_data_set(output)])

    pred = "bad"
    if prediction == 1:
        pred = "good"

    return render_template('index.html',isGood=pred,urlname=output)

if __name__ == "__main__":
    app.run(debug=True)