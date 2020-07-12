
## 1. Importing required modules
  
import numpy as np # linear algebra,
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Matplotlib

import seaborn as sns # Seaborn
from sklearn.preprocessing import StandardScaler # Feature scaling
from sklearn.linear_model import LogisticRegression # Linear models
from sklearn.model_selection import learning_curve, RandomizedSearchCV, ParameterSampler # Searching best parameters
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit # Cross validation
from datetime import datetime # Date functions
from sklearn.svm import SVC # Support vector classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


SEED = 20 # Random state
%matplotlib inline
   
   
## Importing training and test data
train = pd.read_csv("D:/My github/Kaggle-projects/MNSIT-image-classification/datasets/train.csv")
test = pd.read_csv("D:/My github/Kaggle-projects/MNSIT-image-classification/datasets/test.csv")
   
# Shape of test and train set
print(f"Training set = {train.shape}")
print(f"Testing set = {test.shape}")
   
   
print("Training set view...............\n")
print(train.iloc[:,:8].head())
  
  
# Standardising the dataset
scaler = StandardScaler()
scaler.fit(train.iloc[:,1:])
xTrain = scaler.transform(train.iloc[:,1:])
xTest = scaler.transform(test)
yTrain = train.iloc[:,0]
  
    
# Class to perform grid search and cross-validation on the datatset
class evaluation():

    def __init__(self):
        self.X = xTrain
        self.y = yTrain
        
    def getSplits(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
        return sss.split(self.X,self.y)
    
    def doRandomSearch(self, parameters, model, iterations = 5):
        indices =  list(self.getSplits())[0]
        trainIndex, valIndex = indices[0], indices[1]
        scores = []
        start = datetime.now()
        for sample in ParameterSampler(parameters, n_iter=iterations):
            model.set_params(**sample, n_jobs=-1)
            model.fit(self.X[trainIndex], self.y[trainIndex])
            valScore = model.score(self.X[valIndex], self.y[valIndex])
            scores.append([sample, valScore])
        print(f"\nTime taken to do random search = {datetime.now()-start}")
            
        return sorted(scores, key = lambda x:x[1])[-1]
   
    def doCrossValidation(self, model):
        scoring = {'acc': 'accuracy'}
        start = datetime.now()
        scores = cross_validate(
            model, 
            self.X, 
            self.y, 
            scoring=scoring,
            cv=self.getSplits(), 
            return_train_score=False
            )
        
        print(f"\nTotal time taken for cross validation = {datetime.now()-start}")
        print(f"\nValidation set accuracy = {round(scores['test_acc'][0],4)}")
        
              
modelEvaluation = evaluation()


## 4.1. Logistic regression

logModel = LogisticRegression(random_state=SEED, max_iter=1000)
logParams = {
    'penalty': ['l2'],
    'C': [0.0001, 0.001, 0.01, 1,10, 100],
    'max_iter': [1000]
    }

bestParameters = modelEvaluation.doRandomSearch(logParams, logModel, iterations=1)

# Model with best parameters
logClassifier = LogisticRegression(random_state = SEED, **bestParameters[0])

# Cross validation
modelEvaluation.doCrossValidation(logClassifier)


## 4.2. Random forest

# Setting up parameters for random forest
forest = RandomForestClassifier(random_state = SEED)
forestParams = {
    'bootstrap': [True, False],
    'max_depth': range(10, 110, 10),
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': range(1, 5),
    'min_samples_split': range(2, 11),
    'n_estimators': range(200, 2200, 200)

results = modelEvaluation.doRandomSearch(forestParams, forest, iterations=30)


## 4.3. Gradient boosting

# Setting up parameters for gradient boosting
XGB = XGBClassifier()

XGBParams = {
    "learning_rate": np.arange(0.05, 0.30, 0.01) ,
    "max_depth": np.arange(3,30),
    "min_child_weight": [ 1, 3, 5, 7 ],
    "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
    "colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ] 
    }

# Random search
bestParamsXGB = modelEvaluation.doRandomSearch(
    XGBParams,
    XGB, 
    iterations=20
    )

# =============================================================================
# XGB = {'min_child_weight': 3, 
#  'max_depth': 17, 
#  'learning_rate': 0.13, 
#  'gamma': 0.0, 
#  'colsample_bytree': 0.3}
# =============================================================================

# Cross validation
XGBmodel = XGBClassifier(random_state=SEED, **bestParamsXGB[0])
modelEvaluation.doCrossValidation(XGBmodel)


## Predictions and submission

#  Fit the model on the training set
XGBmodel.fit(xTrain, yTrain)

# Make predictions on the test set
XGBpredictions = XGBmodel.predict(xTest)

# Function to write predictions into a csv file
'''
Input: Predictions, filename, ID
Output: Generates a csv file
'''
def pred_to_csv(predictions, file_name):
    
    submissions = pd.DataFrame(columns = ["ImageId", "Label"])
    submissions.ImageId = test.index + 1
    submissions.Label = predictions
    submissions.to_csv('{}.csv'.format(file_name), header=True, index=False)
    
pred_to_csv(XGBpredictions, "XGBmodel")

# Predictions from XGB model give 0.972 accuracy.

ximagelist = train.iloc[6,1:].values.reshape(28,28)
im = plt.imshow(ximagelist, cmap='Greys')
pyplot.show()












  