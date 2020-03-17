# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from xgboost import XGBClassifier

#load in dataset for model training
fullxTrain = pd.read_csv('fullxTrain.csv')
fullyTrain = pd.read_csv('fullyTrain.csv')
fullyTest = pd.read_csv('fullyTest.csv')
fullxTest = pd.read_csv('fullxTest.csv')


#define XGBoost model
ensemble_model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                      n_estimators=100, max_depth=3)

#fit XGBoost ensemble model
ensemble_model = ensemble_model.fit(fullxTrain, fullyTrain)


# Saving model to disk
pickle.dump(ensemble_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

#test a result
test_data = [2,18,10,5,3000,1,12,10,5,6]
test_df = pd.DataFrame(np.array(test_data).reshape(1,10))

#rename columns
x = fullxTrain.columns.tolist()
test_df.columns = x

#predict dummy data
preds = ensemble_model.predict_proba(test_df)
preds = pd.DataFrame(preds)
print(round(100*preds.iloc[0,0],2))

