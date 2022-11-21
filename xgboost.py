import gc
import os
import operator
import warnings
from glob import glob

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def objective(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    watchlist = [(dm1, 'train'), (dm2, 'valid')]
    model = xgb.train(params, dm1, num_round, watchlist, maximize=True, early_stopping_rounds=20, verbose_eval=10)
    pred = model.predict(dm2, ntree_limit=model.best_ntree_limit)
    auc = roc_auc_score(dm2.get_label(), pred)
    del pred,model
    gc.collect()
    print(f"SCORE: {auc}")
    return { 'loss': 1-auc, 'status': STATUS_OK }
def output(model,inputfile,outputfile):
    test=pd.read_csv(inputfile)
    test=test.drop(test.columns[0],axis=1)
    dm_test = xgb.DMatrix(test, feature_names=feature_names)
    predictions_test = model.predict(dm_test,ntree_limit=model.best_ntree_limit)
    Test_df_predictions=pd.DataFrame(data=predictions_test,columns=["isFraud"])
    Test_df_predictions[Test_df_predictions['isFraud']>0.7]=1
    Test_df_predictions[Test_df_predictions['isFraud']<=0.7]=0
    Test_df_predictions.reset_index(inplace=True)
    Test_df_predictions.rename(columns={"index":"Id"},inplace=True)
    Test_df_predictions.head()
    Test_df_predictions.to_csv(outputfile,index=False)


train=pd.read_csv("../input/transaction-fruad/Train_without_Data_balancing.csv")
train=train.drop(train.columns[0],axis=1)
Train_target_df=train.loc[:,"isFraud"]
train.drop("isFraud",axis=1,inplace=True)
x_train, x_test, y_train, y_test = train_test_split(train, Train_target_df, test_size=0.20, random_state=0)

feature_names=x_train.columns
dm1 = xgb.DMatrix(x_train, y_train, feature_names=feature_names)
dm2 = xgb.DMatrix(x_test, y_test, feature_names=feature_names)

space = {
    'n_estimators': hp.quniform('n_estimators', 200, 600, 50),
    'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
    'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
    'alpha' : hp.quniform('alpha', 0, 10, 1),
    'lambda': hp.quniform('lambda', 1, 2, 0.1),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 50, 200, 10),
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': "hist",
    'booster': 'gbtree'
}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=2,
    trials=trials
)

final_pera={
    'alpha': 8.0,
    'colsample_bytree': 0.8,
    'eta': 0.17500000000000002,
    'gamma': 0.6000000000000001,
    'lambda': 1.2000000000000002,
    'max_depth': 10,
    'min_child_weight': 1.0,
    'n_estimators': 550.0,
    'scale_pos_weight': 50.0,
    'subsample': 0.9,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': "hist",
    'booster': 'gbtree'
}
for key,value in best.items():
    final_pera[key]=best[key]
print("best perameters:{}".format(final_pera))

num_round = int(final_pera['n_estimators'])
del final_pera['n_estimators']
watchlist = [(dm1, 'train'), (dm2, 'valid')]
model = xgb.train(final_pera, dm1, num_round, watchlist, maximize=True, early_stopping_rounds=20, verbose_eval=10)
pred = model.predict(dm2, ntree_limit=model.best_ntree_limit)
auc = roc_auc_score(dm2.get_label(), pred)
print(f"SCORE: {auc}")

output(model,'../input/transaction-fruad/Test_without_Data_balancing.csv','./xg_predictions.csv')