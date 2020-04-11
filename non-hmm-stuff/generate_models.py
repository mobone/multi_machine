import yfinance
from ta_indicators import get_ta
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

num_years = 6
def get_data(symbol):

    history = yfinance.Ticker(symbol).history(period=str(num_years)+'y', auto_adjust=False).reset_index()

    history = get_ta(history, volume=True, pattern=False)

    history.columns = map(str.lower, history.columns)
    history['date'] = pd.to_datetime(history['date'])
    #history['return'] = history['close'].pct_change()
    #history['future_price'] = history['close'].shift(-2)
    #history['next_return'] = (history['close'].shift(-3) - history['close']) / history['close']
    history['next_return'] = history['close'].pct_change(2).shift(-2)
    print(history)
    #history['next_return'].shift(-2)


    

    
    pos_cutoff = history[ history['next_return']>0 ]['next_return'].mean()
    neg_cutoff = history[ history['next_return']<0 ]['next_return'].mean()

    #pos_cutoff = history['return'].mean()


    print('pos_cutoff', pos_cutoff, 'neg_cutoff', neg_cutoff)

    #print('pos_cutoff', pos_cutoff)

    history['return_class'] = 0
    history.loc[ history['next_return']>pos_cutoff/2, 'return_class'] = 1
    history.loc[ history['next_return']>pos_cutoff, 'return_class'] = 2
    history.loc[ history['next_return']>pos_cutoff*2, 'return_class'] = 3

    #history['return_class'] = pd.qcut(history['next_return'], q=[0, .33, .66, 1], labels=[0,1,2])
    
    #history.loc[ history['next_return']<neg_cutoff/2, 'return_class'] = -1
    #history.loc[ history['next_return']<neg_cutoff, 'return_class'] = -2

    
    history = history.dropna().reset_index(drop=True)
    print('distribution',history.groupby(by=['return_class'])['open'].count() / len(history))
    print(history.groupby(by=['return_class'])['next_return'].mean())
    return history


history_df = get_data('QQQ')

train = history_df.loc[history_df['date']<'2017-01-01']
test = history_df.loc[history_df['date']>'2017-01-01']
#print(train)
#print(test)
#print(history_df)


X_train = train
y_train = train['return_class']

X_test = test
y_test = test['return_class']

test_cols = list( X_train.columns.drop( [ 'date','open', 'high', 'low', 'close', 'adj close', 'next_return', 'return_class' ] ) )

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mlxtend

#knn = KNeighborsClassifier(n_neighbors=2)
#svc = SVC(gamma='auto')
rfc = RandomForestClassifier(n_estimators=25)
#rfc = ExtraTreesClassifier(n_estimators=10)
sfs1 = SFS(estimator=rfc,
           k_features=5,
           forward=True, 
           floating=True, 
           scoring='accuracy',
           #fixed_features=(5,8),
           cv=2)
#scaler = MinMaxScaler(feature_range = (0, 1))
"""
scaler = StandardScaler()
pipe = Pipeline([('scaler', scaler),
                 ('sfs', sfs1),
                 ('svc', svc)
                ]
                )
"""
pipe = Pipeline([
                 ('sfs', sfs1),
                 ('rfc', rfc)
                ]
                )
param_grid = [
  {
        'sfs__k_features': range(2,9),
        'sfs__estimator__max_depth': range(2,9),
        #'sfs__k_features': range(2,4),
        #'sfs__estimator__gamma': ['auto', 'scale'],
        #'sfs__estimator__shrinking': [True, False],
        #'sfs__estimator__probability': [True, False],
        #'sfs__estimator__decision_function_shape': ['ovo', 'ovr'],
        #'sfs__estimator__C': [.01,.05,.1,.5,1],

        #'sfs__estimator__n_neighbors': [1, 5, 10]
        #'sfs__estimator__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
  }
  ]

gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  n_jobs=15,
                  cv=2,
                  verbose=1,
                  iid=True,
                  refit=True)
#print(gs)
# run gridearch
gs = gs.fit(X_train[test_cols], y_train)
for i in range(len(gs.cv_results_['params'])):
    print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])
print("Best parameters via GridSearch", gs.best_params_)


#c_param = gs.best_params_['sfs__estimator__C']

#feature_indexes = list( gs.best_estimator_.steps[1][1].k_feature_idx_ )


#test_cols = np.array(test_cols)
#best_features = test_cols[ list(feature_indexes) ]
best_features = list(gs.best_estimator_.steps[0][1].k_feature_names_)
print('Best features:', best_features )

#import pdb;
#pdb.set_trace()
#gs.fit(X_train[best_features], y_train)
#svc = SVC(gamma='auto', C=c_param)
"""
test_pipe = make_pipeline(
                            scaler,
                            svc
                        )
test_pipe.fit(X_train[best_features], y_train)
"""

rfc.fit(X_train[best_features], y_train)
#y_pred = test_pipe.predict(X_test[best_features])
y_pred = rfc.predict(X_test[best_features])

X_test['predicted'] = y_pred
X_test['random_predicted'] = np.random.randint(y_pred.min(), y_pred.max()+1, size=len(X_test))
print(X_test)
acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
print('Test set accuracy: %.2f %%' % (acc * 100))
acc = float((y_test == X_test['random_predicted']).sum()) / y_pred.shape[0]
print('Random set accuracy: %.2f %%' % (acc * 100))
print(X_test.groupby(by=['predicted'])['next_return'].mean())
print(X_test.groupby(by=['predicted'])['next_return'].count())
#Best features: ('macd', 'macd_signal', 'beta', 'macdfix_signal', 'ht_trendmode', 'linearreg_slope', 'stddev')
