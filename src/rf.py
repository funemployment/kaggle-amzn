# rf.py
# 06/20/2013
# Random forest implementation for Kaggle amazon challenge

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv(os.path.join(os.getcwd(),'data','train.csv'))
test = pd.read_csv(os.path.join(os.getcwd(),'data','test.csv'))

train_X = train.ix[:,1:]
train_Y = train.ix[:,0]

estim = 50000

clf = RandomForestClassifier(n_estimators = estim, random_state=512, oob_score=True)
clf.fit(train_X, train_Y)

print clf.oob_score_

test_predict = test[['id']]
test_predict['ACTION'] = clf.predict_proba(test.ix[:,1:])[1]

test_predict.to_csv(os.path.join(os.getcwd(),'output','rf_Predict_Proba'+str(estim)+'.csv'),index=False)
