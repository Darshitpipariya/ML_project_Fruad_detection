# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from sklearn.linear_model import RidgeClassifier


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
warnings.filterwarnings('ignore')
train=pd.read_csv("../input/transaction-fruad/Train_without_Data_balancing.csv")
train=train.drop(train.columns[0],axis=1)
Train_target_df=train.loc[:,"isFraud"]
train.drop("isFraud",axis=1,inplace=True)
X=train
y=Train_target_df

#   define model and parameters 
model = RidgeClassifier()
#   define grid search
params = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,1e-3, 1e-2] }
cv = StratifiedKFold(n_splits=5)
random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0,verbose=1)
#  fit the model
random_result = random_search.fit(X, y)
#     Results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
means = random_result.cv_results_['mean_test_score']
stds = random_result.cv_results_['std_test_score']
params = random_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


        
#    output test predictions for model
test=pd.read_csv("../input/transaction-fruad/Test_without_Data_balancing.csv")
test=test.drop(test.columns[0],axis=1)
predictions_test = random_search.predict(test)
Test_df_predictions=pd.DataFrame(data=predictions_test,columns=["isFraud"])
Test_df_predictions.reset_index(inplace=True)
Test_df_predictions.rename(columns={"index":"Id"},inplace=True)
Test_df_predictions.to_csv("./output_predictions.csv",index=False)