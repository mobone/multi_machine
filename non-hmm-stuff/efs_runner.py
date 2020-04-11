import yfinance
from ta_indicators import get_ta
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mlxtend
import time
import namegenerator
import warnings
import sqlite3

conn =  sqlite3.connect('results.db')

warnings.simplefilter("ignore")
num_years = 9
def get_data(symbol, test=None, predictions=None):

    history = yfinance.Ticker(symbol).history(period=str(num_years)+'y', auto_adjust=False).reset_index()

    history = get_ta(history, volume=True, pattern=False)

    history.columns = map(str.lower, history.columns)

    history['date'] = pd.to_datetime(history['date'])

    history['next_return'] = history['close'].pct_change(1).shift(-1)
    
    pos_cutoff = history[ history['next_return']>0 ]['next_return'].mean()
    neg_cutoff = history[ history['next_return']<0 ]['next_return'].mean()
    history['return_class'] = 0
    #history.loc[ history['next_return']<neg_cutoff, 'return_class'] = -1
    #history.loc[ history['next_return']>0, 'return_class'] = 1
    history.loc[ history['next_return']>pos_cutoff/6, 'return_class'] = 1
    #history.loc[ history['next_return']>pos_cutoff*2, 'return_class'] = 3

    history = history.dropna().reset_index(drop=True)
    print('cutoffs %.2f%%' % (pos_cutoff * 100))
    print('distribution',history.groupby(by=['return_class'])['open'].count())
    print(history.groupby(by=['return_class'])['open'].count() / len(history))
    print(history.groupby(by=['return_class'])['next_return'].mean())
    return history

for symbol in ['TQQQ', 'QQQ']:
    history_df = get_data(symbol)



    train = history_df.loc[history_df['date']<'2016-01-01']
    test = history_df.loc[history_df['date']>'2016-01-01']

    train = train[train['date']<'2020-01-01']
    test = test[test['date']<'2020-01-01']
    X_train = train
    y_train = train['return_class']
    X_test = test
    y_test = test['return_class']

    output_df = X_test[ ['date', 'open', 'high', 'low', 'close', 'volume', 'adj close'] ]
    output_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    output_df.to_csv('%s.csv' % symbol, index=False)



test_cols = list( X_train.columns.drop( [ 'date','open', 'high', 'low', 'close', 'adj close', 'next_return', 'return_class' ] ) )

rfc = RandomForestClassifier(n_estimators=10, max_depth=15, random_state=1986)
rfc.fit( X_train[ test_cols ], y_train )
feature_importances = permutation_importance(rfc, X_train[ test_cols ], y=y_train, n_jobs=15, n_repeats=25, random_state=1986)
df = pd.DataFrame( [feature_importances['importances_mean'], test_cols] ).T
df.columns = ['importances', 'feature']
df['importances'] = abs(df['importances'])
df = df.sort_values(by=['importances'])
print('\n\n')


for feature_choices in [10,20,30,40,50]:
    for max_len in [5,10]:
        
        these_choices = df.tail(feature_choices)
        #print(these_choices)
        #print(df)
        test_cols = these_choices['feature'].values
        print(test_cols)
        efs = EFS(
                    estimator=rfc,
                    min_features=3,
                    max_features=max_len,
                    print_progress=False,
                    scoring='accuracy',
                    n_jobs=15,
                    cv=4,
                    
                )
        
        start_time = time.time()
        try:
            efs = efs.fit(X_train[test_cols], y_train)
        except:
            continue
        end_time = time.time()
        #print()
        #print(feature_choices, end_time - start_time)
        best_features = list(efs.best_feature_names_)
        best_score = efs.best_score_
        
        print(feature_choices, max_len, 'time', end_time - start_time,  'score', best_score, 'features', len(best_features), best_features)

        #print()


        rfc.fit(X_train[best_features], y_train)
        #y_pred = test_pipe.predict(X_test[best_features])
        y_pred = rfc.predict(X_test[best_features])
        
        

        X_test.loc[:, 'predicted'] = y_pred
        
        output_df['Close'] = y_pred
        output_df['Low'] = y_pred

        
        

        output_df.to_csv('states.csv', index=False)
        
        #input()
        X_test.loc[:,'random_predicted'] = np.random.randint(y_pred.min(), y_pred.max()+1, size=len(X_test))
        
        #print(X_test)
        acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
        random_acc = float((y_test == X_test['random_predicted']).sum()) / y_pred.shape[0]
        print('Test set accuracy: %.2f%% %.2f%%' %  ((acc * 100), (random_acc * 100))  )
        #print('Random set accuracy: %.2f %%' % (acc * 100))
        means = X_test.groupby(by=['predicted'])['next_return'].mean()

        counts = X_test.groupby(by=['predicted'])['next_return'].count()

        print(means)
        print(counts)
        from strategy import setup_strategy
        #setup_strategy(files, name, strategy)
        files = [
            ('QQQ', 'QQQ.csv'),
            ('TQQQ', 'TQQQ.csv'),
            ('states', 'states.csv')
        ]
        name = namegenerator.gen()
        #print(name)
        result = setup_strategy(files, name).T
        result['name'] = name
        result['qqq_mean'],result['tqqq_mean'] = means.values
        result['qqq_count'],result['tqqq_count'] = counts.values
        print(result)
        result['num_feature_choices'] = feature_choices
        result['max_num_features'] = max_len
        result['time'] = end_time - start_time
        result['features'] = str(best_features)
        result['num_features'] = len(best_features)
        result.to_sql('results', conn, if_exists='append', index=False)
        #input()



