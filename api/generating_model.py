import joblib

# Save Model
model = autoencoder
filename = "./api/models/Completed_model.joblib"
joblib.dump(model, filename)

# Save columns in order
# print("The X_Test", X_test.columns)
# print("The X_train", X_train.columns)
pd.DataFrame(X_train.columns).to_csv("./api/models/feature_list.csv", index = None)

pd.DataFrame(X_test).to_csv("./api/models/X_test.csv", index = None)

# Save data types of train set
pd.DataFrame(X_train.dtypes).reset_index().to_csv("./api/models/data_types.csv", index = None)

pred1 = autoencoder.predict(X_test)
print(pred1)
pred1 = pd.DataFrame(pred1).to_csv('./api/models/data1.csv')

loaded_model = joblib.load(filename)
result = loaded_model.evaluate(X_test, y_test)
print("Evaluation From traind model:", result)