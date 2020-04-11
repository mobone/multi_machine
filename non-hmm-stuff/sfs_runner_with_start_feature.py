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

conn =  sqlite3.connect('sfs_results_50_estimators.db')

warnings.simplefilter("ignore")
num_years = 9
cutoff_divisor = 8
def get_data(symbol, test=None, predictions=None):

    history = yfinance.Ticker(symbol).history(period=str(num_years)+'y', auto_adjust=False).reset_index()

    history = get_ta(history, volume=True, pattern=False)

    history.columns = map(str.lower, history.columns)

    history['date'] = pd.to_datetime(history['date'])

    history['next_return'] = history['close'].pct_change(1).shift(-1)
    
    pos_cutoff = history[ history['next_return']>0 ]['next_return'].mean()/cutoff_divisor
    neg_cutoff = history[ history['next_return']<0 ]['next_return'].mean()
    history['return_class'] = 0
    #history.loc[ history['next_return']<neg_cutoff, 'return_class'] = -1
    #history.loc[ history['next_return']>0, 'return_class'] = 1
    history.loc[ history['next_return']>pos_cutoff, 'return_class'] = 1
    #history.loc[ history['next_return']>pos_cutoff*2, 'return_class'] = 3

    history = history.dropna().reset_index(drop=True)
    print('cutoffs %.2f%%' % (pos_cutoff * 100))
    print('distribution',history.groupby(by=['return_class'])['open'].count())
    print(history.groupby(by=['return_class'])['open'].count() / len(history))
    print(history.groupby(by=['return_class'])['next_return'].mean())
    return history, pos_cutoff

for symbol in ['UPRO', 'SPY']:
    history_df, pos_cutoff = get_data(symbol)



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
    output_df.to_csv('%s_%s.csv' % (symbol, str(cutoff_divisor)), index=False)



test_cols = list( X_train.columns.drop( [ 'date','open', 'high', 'low', 'close', 'adj close', 'next_return', 'return_class' ] ) )

rfc = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=1986)
rfc.fit( X_train[ test_cols ], y_train )
feature_importances = permutation_importance(rfc, X_train[ test_cols ], y=y_train, n_jobs=15, n_repeats=25, random_state=1986)
df = pd.DataFrame( [feature_importances['importances_mean'], test_cols] ).T
df.columns = ['importances', 'feature']
df['importances'] = abs(df['importances'])
df = df.sort_values(by=['importances'])
print('\n\n')

start_features = df.tail(15)
#print(these_choices)
#print(df)
start_features = list(start_features['feature'].values)[::-1]

#test_cols = df.tail(40)['feature'].values

for start_feature in start_features:
    for k_features in range(2,20):
        print(start_feature)
        sfs = SFS(

                    estimator=rfc,
                    k_features = k_features,
                    forward=True,
                    floating=True,
                    verbose=1,
                    scoring='accuracy',
                    n_jobs=15,
                    fixed_features=[start_feature],
                    cv=4,
                    
                )

        
        
        start_time = time.time()
        try:
            sfs = sfs.fit(X_train[test_cols], y_train)
        except:
            continue
        end_time = time.time()
        #print()
        #print(feature_choices, end_time - start_time)
        best_features = list(sfs.k_feature_names_)
        best_score = sfs.k_score_
        
        print(start_feature, k_features, 'time', end_time - start_time,  'score', best_score, 'features', len(best_features), best_features)

        #print()


        rfc.fit(X_train[best_features], y_train)
        #y_pred = test_pipe.predict(X_test[best_features])
        y_pred = rfc.predict(X_test[best_features])
        
        

        X_test.loc[:, 'predicted'] = y_pred
        
        output_df['Close'] = y_pred
        output_df['Low'] = y_pred

        
        

        output_df.to_csv('states_%s.csv' % cutoff_divisor, index=False)
        
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
            ('SPY', 'SPY_%s.csv' % cutoff_divisor),
            ('UPRO', 'UPRO_%s.csv' % cutoff_divisor),
            ('states', 'states_%s.csv' % cutoff_divisor)
        ]
        name = namegenerator.gen()
        #print(name)
        result = setup_strategy(files, name).T
        result['name'] = name
        result['spy_mean'],result['upro_mean'] = means.values
        result['spy_count'],result['upro_count'] = counts.values
        print(result)
        result['start_feature'] = str(start_feature)
        
        result['time'] = end_time - start_time
        result['features'] = str(best_features)
        result['num_features'] = len(best_features)
        result['cutoff_divisor'] = cutoff_divisor
        result['pos_cutoff'] = pos_cutoff
        result.to_sql('results', conn, if_exists='append', index=False)
        #input()



