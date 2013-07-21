# log_reg.py
# 06/20/2013
# Logistic Regression for kaggle amazon (OneHotEncoder from sample.

import os
import numpy as np
import pandas as pd

from scipy import sparse
from sparsesvd import sparsesvd
from sklearn import linear_model

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

train = pd.read_csv(os.path.join(os.getcwd(),'data','train.csv'))
test = pd.read_csv(os.path.join(os.getcwd(),'data','test.csv'))

n = train.shape[0]
comb = np.vstack((train.values[:,1:],test.values[:,1:]))

ohenc, keymap = OneHotEncoder(comb)
train_X = sparse.csc_matrix(ohenc[:n]).todense()
train_Y = train.ix[:,0].values

#ut, s, vt = sparsesvd(train_X, 1000)

#svd_feat = np.dot(ut,diag(s))

reg = linear_model.logistic.LogisticRegression()
reg.fit(train_X,train_Y)
print reg.score(train_X,train_Y)

test_X = sparse.csc_matrix(ohenc[n:]).todense()

test_predict = test[['id']]
test_predict['ACTION'] = reg.predict_proba(test_X)[:,1]

test_predict.to_csv(os.path.join(os.getcwd(),'output','LogisticRegression_Predict.csv'),index=False)
