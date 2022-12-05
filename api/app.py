from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

import joblib
import json
import pandas as pd
import numpy as np
import math
import serverless_wsgi
from datetime import date

from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder


@app.route("/")
def hello_from_root():
    return jsonify(message='Hello from root!')


@app.route("/predict", methods=['POST'])
def hello():

    request_data = json.loads(request.data)
    merchant_number = request_data["merchant_number"]
    card_number = request_data["card_number"]
    amount = request_data["amount"]
    merchant_description = request_data["merchant_description"]
    merchant_state = request_data["merchant_state"]
    
    print("Merchant Number:", merchant_number)
    print("Card Number:", card_number)
    print("Amount:", amount)
    print("Merchant Description:", merchant_description)
    print("Merchant State:", merchant_state)

    todays_date = date.today()
    day = todays_date.day
    month = todays_date.month
    year = todays_date.year

    print("Current date: ", todays_date)

    print("Current year:", year)
    print("Current month:", month)
    print("Current day:", day)


    New_X_test_Dict = {
        "day": [day],
        "month": [month],
        "cardnum": [card_number],
        "merchantnum": [merchant_number],
        "description": [merchant_description],
        "State": [merchant_state],
        "Amount": [amount]    
    }

    print("\n==============================================================")
    print("New_X_test_Dict\n")
    print(New_X_test_Dict)
    print("\n==============================================================\n")

    New_X_test = pd.DataFrame.from_dict(New_X_test_Dict)

    print("\n==============================================================")
    print("New_X_test:\n")
    print( New_X_test )
    print("\n==============================================================\n")

    mm = MinMaxScaler()
    le = LabelEncoder()

    New_X_test_Scaled = New_X_test

    New_X_test_Scaled["merchantnum"] = le.fit_transform(New_X_test['merchantnum'])
    New_X_test_Scaled["description"] = le.fit_transform(New_X_test['description'])
    New_X_test_Scaled["State"] = le.fit_transform(New_X_test['State'])

    # New_X_test["amount_minmax"] = mm.fit_transform(New_X_test['Amount'].values.reshape(-1,1))

    New_X_test_Scaled['Cardnum_minmax'] = mm.fit_transform(New_X_test_Scaled['cardnum'].values.reshape(-1,1))

    New_X_test_Scaled['amount_log'] = np.log(New_X_test.Amount + 0.0001)
    New_X_test_Scaled['Cardnum_log'] = np.log(New_X_test.cardnum + 0.0001)
    New_X_test_Scaled['Merchnum_log'] = np.log(New_X_test_Scaled.merchantnum + 0.0001)
    New_X_test_Scaled['Merch description_log'] = np.log(New_X_test_Scaled.description + 0.0001)
    New_X_test_Scaled['Merch state_log'] = np.log(New_X_test_Scaled.State + 0.0001)

    print("\n==============================================================")
    print("New_X_test_Scaled:\n")
    print( New_X_test_Scaled )
    print("\n==============================================================\n")
 

    feature_list = pd.read_csv("./models/feature_list.csv")
    feature_list = pd.Index(list(feature_list["0"]))

    # X_test = pd.read_csv("./models/X_test.csv")
    X_test = New_X_test_Scaled

    print("\n==============================================================")
    print("X_test:\n")
    print(X_test)
    print("\n==============================================================\n")
    
    add_cols = list(feature_list.difference(X_test.columns))

    print("add_cols",add_cols)

    drop_cols = list(X_test.columns.difference(feature_list))

    print("drop_cols", drop_cols)

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
    
    print("X_test 2", X_test)
    
    # Making Predictions
    filename = "./models/Completed_model.joblib"

    loaded_model = joblib.load(filename)
    
    prediction = loaded_model.predict(X_test) 
    print(prediction)
    print(prediction[0])
    average = sum(prediction[0]) / len(prediction[0])
    print(average)
    average = (math.ceil(average*100)/100) 
    print(average)
    threshold = 5
    is_fraud = False
    success = True
    message = "Transaction Successful"
    if(average>threshold):
        is_fraud = True
        success = False
        message = "Transaction Failed. Suspected fraud"
    
    response = jsonify(success=success, is_fraud=is_fraud, message=message, prediction=prediction.tolist())
    response.headers.add("Access-Control-Allow-Origin","*")
    response.headers.add("Access-Control-Allow-Headers","*")
    return response


@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)


def handler(event, context):
    return serverless_wsgi.handle_request(app, event, context)