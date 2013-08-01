# log_reg.py
# 06/20/2013
# updated 07/21/2013 - use more sklearn stuff and add features
# Logistic Regression for kaggle amazon

import os
import numpy as np
import pandas as pd
import itertools as it

from collections import Counter
from operator import add

from scipy import sparse
from sklearn import preprocessing, linear_model
from sklearn.cross_validation import KFold
from sklearn.metrics import auc_score

def run_cv(x,y,reg,cv):
     ''' returns mean AUC for this reg using cv splits.'''
     scores = []      
     for sp in cv:
          reg.fit(x[sp[0],:],y[sp[0]])
          scores.append(auc_score(y[sp[1]],reg.predict_proba(x[sp[1],:])[:,1]))
     return np.mean(scores)

def uniquecombtwo(x,y):
     return int('{:06d}{:06d}'.format(x,y))

def pair_columns(df):
     n = df.shape[1]
     for comb in it.combinations(df.columns,2):
          if ('ROLE_TITLE' in comb and 'ROLE_FAMILY' in comb):
               continue
          df[comb[0]+comb[1]] = df.apply(lambda x: uniquecombtwo(x[comb[0]],x[comb[1]]),axis=1)
     return df.ix[:,n:]

def uniquecombthree(x,y,z):
     return int('{:06d}{:06d}{:06d}'.format(x,y,z))

def triple_columns(df):
     n = df.shape[1]
     for comb in it.combinations(df.columns,3):
          if('ROLE_TITLE' in comb and 'ROLE_FAMILY' in comb):
               continue
          df[comb[0]+comb[1]+comb[2]] = df.apply(lambda x: uniquecombthree(x[comb[0]],x[comb[1]],x[comb[2]]),axis=1)
     return df.ix[:,n:]

def uniquecombfour(a,b,c,d):
     return hash(int('{:06d}{:06d}{:06d}{:06d}'.format(a,b,c,d)))

def quad_columns(df):
     n = df.shape[1]
     for comb in it.combinations(df.columns,4):
          if('ROLE_TITLE' in comb and 'ROLE_FAMILY' in comb):
               continue
          df[comb[0]+comb[1]+comb[2]+comb[3]] = df.apply(lambda x: uniquecombfour(x[comb[0]],x[comb[1]],x[comb[2]],x[comb[3]]),axis=1)
     return df.ix[:,n:]

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

# ROLE_CODE == ROLE_TITLE
train = train.drop('ROLE_CODE',axis=1)
test = test.drop('ROLE_CODE',axis=1)

train_x = train.values[:,1:]
test_x = test.values[:,1:]
train_y = train.values[:,0]

N = train_x.shape[0]

x_single = np.vstack((train_x, test_x))

train_x_pairs = pair_columns(train.ix[:,1:])
test_x_pairs = pair_columns(test.ix[:,1:])

train_x_trips = triple_columns(train.ix[:,1:])
test_x_trips = triple_columns(test.ix[:,1:])

train_x_quads = quad_columns(train.ix[:,1:])
test_x_quads = quad_columns(test.ix[:,1:])

cur_best = 0
pairs = [2]
trips = [2,3,4]
quads = [15,17,20]

for quad_thresh in quads:
     x_quads = np.vstack((train_x_quads,test_x_quads))

     good_cols = []
     for i in range(x_quads.shape[1]):
          i_count = Counter(x_quads[:N,i])
          if(i_count.most_common(1)[0][1] >= quad_thresh):
               good_cols.append(i)
          def_val = i_count.most_common(len(i_count))[-1][0]
          for j in range(x_quads.shape[0]):
               if(i_count[x_quads[j,i]]<quad_thresh):
                    x_quads[j,i] = def_val

     x_quads = x_quads[:,good_cols]

     for trip_thresh in trips:
          x_trips = np.vstack((train_x_trips, test_x_trips))

          good_cols = []
          for i in range(x_trips.shape[1]):
               i_count = Counter(x_trips[:N,i])
               if(i_count.most_common(1)[0][1] >= trip_thresh):
                    good_cols.append(i)
               def_val = i_count.most_common(len(i_count))[-1][0]
               for j in range(x_trips.shape[0]):
                    if(i_count[x_trips[j,i]]<trip_thresh):
                         x_trips[j,i] = def_val

          x_trips = x_trips[:,good_cols]

          for pair_thresh in pairs:
               x_pairs = np.vstack((train_x_pairs, test_x_pairs))
     

          #pair_counter = reduce(add, (Counter(x_pairs[:,i]) for i in range(x_pairs.shape[1])))
               good_cols = []
               for i in range(x_pairs.shape[1]):
                    i_count = Counter(x_pairs[:N,i])
                    if(i_count.most_common(1)[0][1] >= pair_thresh):
                         good_cols.append(i)
                    def_val = i_count.most_common(len(i_count))[-1][0]
                    for j in range(x_pairs.shape[0]):
                         if(i_count[x_pairs[j,i]]<pair_thresh):
                              x_pairs[j,i] = def_val

               x_pairs = x_pairs[:,good_cols]

               x_sp = np.hstack((x_single, x_pairs, x_trips, x_quads))

               print "Encoding"
               enc,keymap = OneHotEncoder(x_sp)

               train_x_enc = enc[:N,:]
               test_x_enc = enc[N:,:]

          # 10-fold cross val
               param_grid = it.product(['l2'],[0.1,0.3,1])
               #param_grid = it.product(['l2'],[0.1])
               for params in param_grid:
                    cv = KFold(train_x_enc.shape[0],n_folds=10,shuffle=True,random_state=512)
                    print quad_thresh, trip_thresh, pair_thresh, params
                    reg = linear_model.LogisticRegression(penalty=params[0],C=params[1],random_state=512)
                    result = run_cv(train_x_enc,train_y,reg,cv)
                    print result
                    if result > cur_best:
                         cur_best = result
                         best_params = params
                         best_pair = pair_thresh
                         best_trip = trip_thresh
                         best_quad = quad_thresh

#reg.fit(train_x_enc, train_y)
#test_predict = test[['id']]
#test_predict['ACTION'] = reg.predict_proba(test_x_enc)[:,1]

#test_predict.to_csv(os.path.join(os.getcwd(),'output','LogisticRegression_Predict_2_4_10.csv'),index=False)
