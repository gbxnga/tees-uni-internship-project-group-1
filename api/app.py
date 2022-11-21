from flask import Flask, jsonify, make_response

app = Flask(__name__)

import joblib
import json
import pandas as pd
import numpy as np


@app.route("/")
def hello_from_root():
    return jsonify(message='Hello from root!')


@app.route("/predict", methods=['POST'])
def hello():

    feature_list = pd.read_csv("./models/feature_list.csv")
    feature_list = pd.Index(list(feature_list["0"]))

    X_test = pd.read_csv("./models/X_test.csv")
    print(X_test.columns)
    add_cols = list(feature_list.difference(X_test.columns))

    drop_cols = list(X_test.columns.difference(feature_list))

    for col in add_cols:
        X_test[col] = np.nan

    for col in drop_cols:
        X_test = X_test.drop(col, axis = 1)
     
    # reorder columns
    X_test = X_test[feature_list]
    
    types = pd.read_csv("./models/data_types.csv")

    print("types", types)
    print("X_test", X_test)
    for i in range(len(types)):
        X_test[types.iloc[i,0]] = X_test[types.iloc[i,0]].astype(types.iloc[i,1])
    
    
    # Making Predictions
    filename = "./models/Completed_model.joblib"

    loaded_model = joblib.load(filename)
    
    prediction = loaded_model.predict(X_test) 
    print(prediction)
    return jsonify(prediction=prediction.tolist())


@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)