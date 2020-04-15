from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
from mlxtend.feature_extraction import PrincipalComponentAnalysis
import yfinance
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import namegenerator
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

def generate_model(features, n_subsets, n_components, lookback, name):
    
    train = pd.read_csv('./datasets/train.csv')
    test = pd.read_csv('./datasets/test.csv')

    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])
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
                train['state'] = pipe_pca.predict(train[ features ])
                results = pd.DataFrame(train.groupby(by=['state'])['return'].mean().sort_values())
                results['new_state'] = list(range(n_components))
                results.columns = ['mean', 'new_state']
                results = results.reset_index()
                results['name'] = int_name
                int_name = int_name + 1
                pipelines.append( [pipe_pca, results] )
            except Exception as e:
                #print('make trained pipelines exception', e)
                pass
        
        return pipelines


    def run_pipeline(pipelines, test):
        for i in range(lookback,len(test)):
            this_test = test.iloc[ i - lookback : i]
            today = this_test[-1:]
            max_score = -np.inf
            for pipeline, train_results in pipelines:
                test.loc[today.index, 'state'] = 0
                test.loc[today.index, 'model_used'] = 'None'
                try:
                    test_score = np.exp( pipeline.score( this_test[ features ]) / len(this_test) ) * 100
                    if test_score>max_score:
                        state = pipeline.predict( this_test[ features ] )[-1:][0]
                        state = int(train_results[train_results['state']==state]['new_state'])
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


