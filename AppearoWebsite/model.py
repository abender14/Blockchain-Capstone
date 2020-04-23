# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import gzip

#load in dataset for model training
fullxTrain = pd.read_csv('fullxTrain.csv')
fullyTrain = pd.read_csv('fullyTrain.csv')
fullyTest = pd.read_csv('fullyTest.csv')
fullxTest = pd.read_csv('fullxTest.csv')


#define XGBoost model
ensemble_model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                      n_estimators=100, max_depth=3)

ada_model = AdaBoostClassifier(n_estimators=100)

gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=10, random_state=0)

#define random forest model
rf = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=1)

#fit XGBoost ensemble model
ensemble_model = ensemble_model.fit(fullxTrain, fullyTrain)

ada_model = ada_model.fit(fullxTrain, fullyTrain)

#fit model to training data
rf_model = rf.fit(fullxTrain, fullyTrain)

#fit model to training data
gb_model = gb_model.fit(fullxTrain, fullyTrain)


knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model = knn_model.fit(fullxTrain, fullyTrain)

#define neural network model
ann_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

#fit model to training dataset
ann_model.fit(fullxTrain, fullyTrain.values.ravel())

# Saving model to disk
#ensemble_model.save_model('model.json')

#dump model using joblib
#joblib_file = "joblib_ML_Model2.json"
#joblib.dump(gb_model, joblib_file)

#load ml model using joblib
#joblib_ML_model = joblib.load(joblib_file)

#output model parameters
#joblib_ML_model

# Saving model to disk
pickle.dump(ensemble_model, open('model.pkl','wb'))
ensemble_model.dump_model('xgb_model.pkl')
pickle.dump(rf_model, open('model2.pkl','wb'))
pickle.dump(gb_model, open('model3.pkl','wb'))
pickle.dump(ada_model, open('model4.pkl','wb'))
pickle.dump(knn_model, open('model5.pkl','wb'))
pickle.dump(ann_model, open('model6.pkl','wb'))
# Loading model to compare the results
#model = pickle.load(open('model3.pkl','rb'))

#load saved model
#xgb2 = joblib.load(joblib_file)

# Loading model to compare the results
#model = open('model.json','rb')
model = pickle.load(open('model6.pkl', 'rb'))


#test a result
test_data = [.1,22,0,0.25,300,0,4,0,0,0]
test_df = pd.DataFrame(np.array(test_data).reshape(1,10))

#rename columns
x = fullxTrain.columns.tolist()
test_df.columns = x

#predict dummy data
preds = model.predict_proba(test_df)
preds = pd.DataFrame(preds)
print(round(100*preds.iloc[0,1],2))
