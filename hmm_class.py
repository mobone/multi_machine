from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
from mlxtend.feature_extraction import PrincipalComponentAnalysis
import yfinance
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import namegenerator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import numpy as np
from time import sleep
import warnings
from multiprocessing import Pool, cpu_count
from utils import get_data, plot
from hmm_strategy import setup_strategy
from sklearn.ensemble import ExtraTreesRegressor
import io
from itertools import product
import sqlite3
from itertools import combinations
from random import shuffle, randint
from rq import Queue
from redis import Redis
warnings.simplefilter('ignore')

def generate_model(features, n_subsets, n_components, lookback, with_rfc, include_covid, name):
    
    train = pd.read_csv('./datasets/train.csv')
    test = pd.read_csv('./datasets/test.csv')

    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    if include_covid == False:
        train = train[train['date']<'2020-01-01']
        test = test[test['date']<'2020-01-01']

    # decision tree stuff
    cutoff_divisor = 8
    pos_cutoff = train[ train['next_return']>0 ]['next_return'].mean()/cutoff_divisor

    test['return_class'] = 0
    test.loc[ test['next_return']>pos_cutoff, 'return_class'] = 1

    train['return_class'] = 0
    train.loc[ train['next_return']>pos_cutoff, 'return_class'] = 1

    def train_decision_tree(train, y_train):
        #print('making deicision tree')
        #print(train)
        #print(y_train)
        rfc_features = list(train.columns)
        if 'return' in rfc_features:
            rfc_features.remove('return')

        rfc = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=1986)
        rfc.fit( train[rfc_features], y_train )
        return rfc

    def get_trained_pipelines(train):
        train_dfs = np.array_split(train, n_subsets)
        int_name = 0
        pipelines = []
        
        for train_subset in train_dfs:
            try:
                pipe_pca = make_pipeline(StandardScaler(),
                            PrincipalComponentAnalysis(n_components=n_components),
                            GMMHMM(n_components=n_components, covariance_type='full', n_iter=150, random_state=7),
                            )
                pipe_pca.fit(train_subset[ features ])

                if with_rfc:
                    rfc = train_decision_tree(train_subset[ features ], train_subset['return_class'])
                else:
                    rfc = None

                train['state'] = pipe_pca.predict(train[ features ])
                results = pd.DataFrame(train.groupby(by=['state'])['return'].mean().sort_values())
                results['new_state'] = list(range(n_components))
                results.columns = ['mean', 'new_state']
                results = results.reset_index()
                results['name'] = int_name
                int_name = int_name + 1
                
                pipelines.append( [pipe_pca, results, rfc] )
                
            except Exception as e:
                #print('make trained pipelines exception', e)
                pass
        
        return pipelines


    def run_pipeline(pipelines, test):
        
        test.loc[:, 'state'] = None
        test.loc[:, 'rfc_state'] = None
        test.loc[:, 'model_used'] = 'None'
        for i in range(lookback,len(test)+1):
            this_test = test.iloc[ i - lookback : i]
            today = this_test[-1:]
            max_score = -np.inf
            for pipeline, train_results, rfc in pipelines:
                
                try:
                    test_score = np.exp( pipeline.score( this_test[ features ]) / len(this_test) ) * 100

                    if test_score>max_score:
                        state = pipeline.predict( this_test[ features ] )[-1:][0]
                        if with_rfc:
                            rfc_state = rfc.predict( this_test[ features ])[-1:][0]
                            test.loc[today.index, 'rfc_state'] = rfc_state
                        
                        state = int(train_results[train_results['state']==state]['new_state'])
                        #print(state, rfc_state)
                        
                        if with_rfc == True and rfc_state>0:
                            test.loc[today.index, 'state'] = state
                        elif with_rfc == True and rfc_state==0:
                            test.loc[today.index, 'state'] = 0
                        elif with_rfc == False:
                            test.loc[today.index, 'state'] = state
                        
                        test.loc[today.index, 'model_used'] = train_results['name'].values[0]
                        max_score = test_score
                except Exception as e:
                    #print('this exception', e)
                    continue
        
        test = test.dropna(subset=['state'])
        models_used = str(test['model_used'].unique())
        num_models_used = len(test['model_used'].unique())

        return test, models_used, num_models_used



    pipelines = get_trained_pipelines(train)

    test, models_used, num_models_used = run_pipeline(pipelines, test)

    #backtest_results, backtest_plot get_backtest(test)

    #print(test.groupby(by='state')['return'].mean())
    #print(test.groupby(by='state')['next_return'].mean())
    #print(test.groupby(by='state')['next_return'].std())
    
    #states_plot = plot(test, name=name, show=False)

    return test, models_used, num_models_used

"""
name = namegenerator.gen()
test, models_used, num_models_used = generate_model(['mom', 'rsi', 'return'], 5, 3, 200, True, name)
print(test)
test.to_csv('test.csv')
"""