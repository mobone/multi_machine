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
from utils import get_data, plot, run_feature_importances
from hmm_strategy import setup_strategy

import io
from itertools import product
import sqlite3
from itertools import combinations
from random import shuffle, randint
from rq import Queue, Connection
from redis import Redis
from rq.job import Job
from hmm_class import generate_model
import time
import hashlib

class run_machine():
    def __init__(self, params):
        self.params  = params
        starting_feature, feature_choices, n_subsets, n_components, lookback, with_rfc = params
        self.starting_feature = starting_feature
        self.feature_choices = feature_choices
        self.n_subsets = n_subsets
        self.n_components = n_components
        self.lookback = lookback
        self.with_rfc = with_rfc

        self.features = [starting_feature]
        self.conn = sqlite3.connect('results.db')

        #TODO: cover cases where all states are none

    def lookup_result(feature_hash):
        previously_found = None
        try:
            sql = 'select * from results where feature_hash=="%s"' % feature_hash
            previously_found = pd.read_sql(sql, self.conn)
        except Exception as e:
            pass

        return previously_found


    def run(self):
        while len(self.features)<21:
            self.results = pd.DataFrame()

            self.jobs = []
            self.create_jobs()
            self.get_redis_results()


    def create_jobs(self):
        q = Queue(connection=Redis( host='192.168.1.127' ))
        for new_feature in self.feature_choices:
            if new_feature in self.features:
                continue

            test_features = self.features + [new_feature]

            feature_hash = str(params)+'_'+str(test_features)
            
            feature_hash = hashlib.md5(feature_hash.encode()).hexdigest()

            previous_result = self.lookup_result(feature_hash)

            if previous_result is None:
                name = namegenerator.gen()

                job_name = name+'__'+str(test_features)+'__'+feature_hash
                self.results = self.results.append( [ [str(test_features), feature_hash, None, None, None] ] )
                print('creating job', job_name)
                # TODO: possibly use different queues for 
                job = q.enqueue(generate_model, args = (test_features, n_subsets, n_components, lookback, name, ), job_timeout='12h',  result_ttl=86400 )

                self.jobs.append( (job.id, job_name) )
            else:
                sharpe_ratio = float(previously_found['sharpe_ratio'])
                cum_returns = float(previously_found['cum_returns'])
                print('found result', job_name, sharpe_ratio, cum_returns)
                self.results = self.results.append( [ [str(test_features), feature_hash, sharpe_ratio, cum_returns, 'finished'] ] )
        self.results.columns = ['features', 'feature_hash', 'sharpe_ratio', 'cum_returns', 'job_status']


    def get_redis_results(self):
        best_sharpe_ratio = -np.inf
        while True:
            redis_con = Redis( host='192.168.1.127' )

            for job_id, job_name in jobs:
                name, features, feature_hash = job_name.split('__')
                job = Job.fetch(job_id, connection = redis_con)

                job_status = job.get_status()

                if job != 'finished' and job != 'failed':
                    continue

                # if backtest has already been completed, continue
                if results_df.loc[results_df['features']==str(features), 'job_status'].values[0] != None:
                    continue

                test_with_states, models_used, num_models_used = job.result

                backtest_results = get_backtest(test_with_states, feature_hash, features, params, models_used, num_models_used, name=name, show_plot=False)
                print(backtest_results)
                sharpe_ratio = float(backtest_results['sharpe_ratio'])
                cum_returns = float(backtest_results['cum_returns'])

                self.results.loc[self.results['features']==str(features), 'sharpe_ratio'] = sharpe_ratio
                self.results.loc[self.results['features']==str(features), 'cum_returns'] = cum_returns
                self.results.loc[self.results['features']==str(features), 'job_status'] = job_status
                
                best_features = self.results[ self.results['sharpe_ratio'] == self.results['sharpe_ratio'].max() ]['features'].values[0]
                if 'win_rate' in backtest_results.columns:
                    backtest_results.to_sql('results', conn, if_exists='append')
                if sharpe_ratio > best_sharpe_ratio:
                    if sharpe_ratio > 1:
                        plot(test_with_states, name=name, show=False)
                    best_sharpe_ratio = sharpe_ratio
                    best_features = features
            
            print(results_df)
            if len(results_df[results_df['sharpe_ratio'].isnull()]):
                sleep(10)
            else:
                break
            
        
        print('found best features', best_features)
        self.features = eval(best_features)



    


if __name__ == '__main__':
    
    history_df = get_data('SPY', period='20y')

    train = history_df.loc[history_df['date']<'2015-01-01']
    test = history_df.loc[history_df['date']>'2015-01-01']

    train.to_csv('./datasets/train.csv', index=False)
    test.to_csv('./datasets/test.csv', index=False)
    
    feature_choices, top_starting_features = run_feature_importances(train, n_total_features=20)
    """
    n_subsets = [3,5,10]
    n_components = [3,4]
    lookback = [50,100,150,200]
    with_rfc = [True, False]

    params = list(product( top_starting_features, n_subsets, n_components, lookback, with_rfc ))

    shuffle(params)
    """

    params = ['mom', feature_choices, 5, 4, 200, True ]
    run_machine( params )