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
from utils import get_data, plot, run_feature_importances, get_backtest
from hmm_strategy import setup_strategy
from rq.registry import FinishedJobRegistry
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

# TODO: Try using hmm state as input for rfc
class run_machine():
    def __init__(self, params, feature_choices):
        print('sleeping')
        print(params)
        sleep(randint(0,30))
        print('starting')
        self.params  = params
        starting_feature, n_subsets, n_components, lookback, with_rfc, include_covid = self.params
        self.starting_feature = starting_feature
        self.feature_choices = feature_choices
        self.n_subsets = n_subsets
        self.n_components = n_components
        self.lookback = lookback
        self.with_rfc = with_rfc
        self.include_covid = include_covid

        self.features = ['return', starting_feature]
        self.conn = sqlite3.connect('results.db')

        #TODO: cover cases where all states are none

        self.run()

    def lookup_result(self, feature_hash):
        previous_result = None
        try:
            #print('looking up', feature_hash)
            sql = 'select * from results where feature_hash=="%s"' % feature_hash
            previous_result = pd.read_sql(sql, self.conn)
            if previous_result.empty:
                previous_result = None
        except Exception as e:
            pass

        return previous_result


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
            #print(self.params, test_features)

            feature_hash = str(self.params)+'_'+str(test_features)
            #print(feature_hash)
            
            feature_hash = hashlib.md5(feature_hash.encode()).hexdigest()

            previous_result = self.lookup_result(feature_hash)

            if previous_result is None:
                name = namegenerator.gen()

                job_name = name+'__'+str(test_features)+'__'+feature_hash
                
                print('creating job', job_name)
                # TODO: possibly use different queues for each simulation run
                job_args = (test_features, 
                            self.n_subsets, 
                            self.n_components, 
                            self.lookback, 
                            self.with_rfc, 
                            self.include_covid,
                            name, )
                job = q.enqueue(generate_model, args = job_args, job_timeout='6h',  result_ttl=3600 )
                #TODO: change to nan and check for isnan in redis results
                self.results = self.results.append( [ [str(test_features),  str(self.params), feature_hash, job.id, np.inf, np.inf, job.get_status()] ] )
                self.jobs.append( (job.id, job_name) )
            else:
                #print(previous_result)
                sharpe_ratio = float(previous_result['sharpe_ratio'])
                cum_returns = float(previous_result['cum_returns'])
                #print('found result', job_name, sharpe_ratio, cum_returns)
                self.results = self.results.append( [ [str(test_features), str(self.params), feature_hash, 'previous', sharpe_ratio, cum_returns, 'previous'] ] )
        self.results.columns = ['features', 'params', 'feature_hash', 'job_id', 'sharpe_ratio', 'cum_returns', 'job_status']
        self.results = self.results.reset_index(drop=True)


    def get_redis_results(self):
        best_sharpe_ratio = -np.inf
        while True:
            redis_con = Redis( host='192.168.1.127' )
            #q = Queue(connection=redis_con)
            #registry = FinishedJobRegistry(queue=q)
            for job_id, job_name in self.jobs:
                name, features, feature_hash = job_name.split('__')
                job = Job.fetch(job_id, connection = redis_con)

                job_status = job.get_status()
                self.results.loc[self.results['features']==str(features), 'job_status'] = job_status
                #print(job_id, job_status, type(job_status))
                if job_status == 'started' or job_status == 'queued':
                    continue
                
                # if backtest has already been completed, continue
                if self.results[self.results['feature_hash']==feature_hash]['sharpe_ratio'].values[0] != np.inf:
                    continue
                try:
                    test_with_states, models_used, num_models_used = job.result
                except Exception as e:
                    print(e)
                    continue
                print('running backtest')
                backtest_results = get_backtest(test_with_states, feature_hash, features, self.params, models_used, num_models_used, name=name, show_plot=False)
                print(backtest_results)
                sharpe_ratio = float(backtest_results['sharpe_ratio'])
                cum_returns = float(backtest_results['cum_returns'])

                self.results.loc[self.results['features']==str(features), 'sharpe_ratio'] = sharpe_ratio
                self.results.loc[self.results['features']==str(features), 'cum_returns'] = cum_returns
                #self.results.loc[self.results['features']==str(features), 'job_status'] = job_status
                
                best_features = self.results[ self.results['sharpe_ratio'] == self.results['sharpe_ratio'].max() ]['features'].values[0]
                # remove job
                #if job_status == 'finished':
                #    registry.remove(job_id)
                # todo: store failed jobs
                if 'win_rate' in backtest_results.columns:
                    backtest_results.to_sql('results', self.conn, if_exists='append')
                if sharpe_ratio > best_sharpe_ratio:
                    if sharpe_ratio > 1:
                        plot(test_with_states, name=name, show=False)
                    best_sharpe_ratio = sharpe_ratio
                    best_features = features
            
            print(self.results)
            num_queued = len(self.results[self.results['job_status']=='queued'])
            num_started = len(self.results[self.results['job_status']=='started'])
            if (num_queued + num_started)>0:
                print('waiting for', num_queued, num_started)
                sleep(10)
            else:
                break
            
        
        print('found best features', best_features)
        self.features = eval(best_features)



def runner_method(params_with_features):
    params, feature_choices = params_with_features
    run_machine(params, feature_choices)


if __name__ == '__main__':
    
    history_df = get_data('SPY', period='20y')

    train = history_df.loc[history_df['date']<'2015-01-01']
    test = history_df.loc[history_df['date']>'2015-01-01']

    train.to_csv('./datasets/train.csv', index=False)
    test.to_csv('./datasets/test.csv', index=False)
    
    feature_choices, top_starting_features = run_feature_importances(train, n_total_features=45)
    
    n_subsets = [3,5,10,15,20]
    # todo: test and work on n_components 3
    n_components = [4]
    lookback = [50,100,150,200]
    with_rfc = [True, False]
    include_covid = [True, False]

    params = list(product( top_starting_features, n_subsets, n_components, lookback, with_rfc, include_covid ))

    shuffle(params)
    
    params_with_features = []
    for param in params:
        params_with_features.append([param, feature_choices])

    #params = ['mom', feature_choices, 5, 4, 200, True ]
    #run_machine( params )

    p = Pool(8)
    p.map(runner_method, params_with_features)

    #runner_method(params_with_features[0])