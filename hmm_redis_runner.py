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
from rq import Queue, Connection
from redis import Redis
from rq.job import Job
from hmm_class import generate_model
import time
import hashlib

warnings.simplefilter('ignore')


def run_decision_tree(train):
    test_cols = list(train.columns.drop(['date','open', 'high', 'low', 'close', 'return', 'next_return']))
    # get features
    clf = ExtraTreesRegressor(n_estimators=150, random_state=42)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances')
    
    feature_choices = list(df['feature'].tail(45).values)

    preset_features = ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                        'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                        'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ]

    feature_choices = list(set( feature_choices + preset_features ))
    print('number of feature choices')
    print(len(feature_choices))
    
    top_starting_features = list(df.sort_values(by='importances').tail(15)['feature'].values)[::-1]
    return feature_choices, top_starting_features


history_df = get_data('SPY', period='15y')

train = history_df.loc[history_df['date']<'2015-01-01']
test = history_df.loc[history_df['date']>'2015-01-01']

train.to_csv('./datasets/train.csv', index=False)
test.to_csv('./datasets/test.csv', index=False)

feature_choices, top_starting_features = run_decision_tree(train)


def get_backtest(test, feature_hash, features, params, models_used, num_models_used, name=None, show_plot=False):
    import dateutil.parser

    starting_feature, n_subsets, n_components, lookback = params

    symbols = ['SPY', 'SSO', 'UPRO']
    files = []
    for symbol in symbols:
        filename = './datasets/%s_%s.csv' % (symbol, name)
        
        df = yfinance.Ticker(symbol).history(period='10y', auto_adjust=False).reset_index()
            
            
        df = df[df['Date']>=test['date'].head(1).values[0]]
        df.to_csv(filename, index=False)

        files.append( (symbol, filename) )
    
    test_df = test[ ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume', 'state'] ]
    test_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'State']            
    test_df['Low'] = test_df['State']
    test_df['Close'] = test_df['State']
    
    test_df.to_csv('./predictions/%s.csv' % name)
    
    
    files.append( (name, './predictions/%s.csv'% name) )
        
    backtest_results = setup_strategy(files, name, show_plot=show_plot).T
    
    backtest_results['avg_per_day'] = float(backtest_results['cum_returns']) / float(len(test))
    backtest_results['models_used'] = models_used
    backtest_results['num_models_used'] = num_models_used
    backtest_results['name'] = name
    backtest_results['start_feature'] = starting_feature
    backtest_results['features'] = str(features)
    backtest_results['num_features'] = len(eval(features))
    
    backtest_results['n_subsets'] = n_subsets
    backtest_results['n_components'] = n_components
    backtest_results['lookback'] = lookback
    backtest_results['feature_hash'] = feature_hash

    return backtest_results


def runner(params):
    sleep_time = randint(0,30)
    print('sleeping for %s seconds before starting' % sleep_time)
    sleep(sleep_time)
    print('starting process!')
    conn =  sqlite3.connect('results.db')
    print(params)
    starting_feature, n_subsets, n_components, lookback = params
    shuffle(feature_choices)
    features = [starting_feature]
    

    while len(features)<16:
        
        results_df = pd.DataFrame()
        
        jobs = []
        q = Queue(connection=Redis( host='192.168.1.127' ))            
        for new_feature in feature_choices:
            if new_feature in features:
                continue

            test_features = features + [new_feature]
            
            test_features.sort()
            
            feature_hash = str(params)+'_'+str(test_features)
            
            feature_hash = hashlib.md5(feature_hash.encode()).hexdigest()
            
            try:
                sql = 'select * from results where feature_hash=="%s"' % feature_hash
                already_found_df = pd.read_sql(sql, conn)
            except Exception as e:
                print(e)
                already_found_df = pd.DataFrame()

            if already_found_df.empty:
                
        
        
                name = namegenerator.gen()

                job_name = name+'__'+str(test_features)+'__'+feature_hash
                results_df = results_df.append( [ [str(test_features), feature_hash, None, None] ] )
                
                #print('creating job', job_name)
                
                print('creating job', feature_hash)
                job = q.enqueue(generate_model, args = (test_features, n_subsets, n_components, lookback, name, ), job_timeout='12h',  result_ttl=86400 )
                #job = Job.create(generate_model, id=feature_hash, args = (test_features, n_subsets, n_components, lookback, name, ), timeout='12h',  result_ttl=86400 )
                #q.enqueue(job)
                #sleep(.25)
                
                jobs.append( (job.id, job_name) )
            else:
                #print('job already found')
                sharpe_ratio = float(already_found_df['sharpe_ratio'])
                cum_returns = float(already_found_df['cum_returns'])
                results_df = results_df.append( [ [str(test_features), feature_hash, sharpe_ratio, cum_returns] ] )
            

            
        
        results_df.columns = ['features', 'feature_hash', 'sharpe_ratio', 'cum_returns']
        sleep(5)
        start_time = time.time()
        
        
        best_sharpe_ratio = -np.inf
        while True:
            redis_con = Redis( host='192.168.1.127' )
            #q = Queue(connection=redis_con)
            for job_id, job_name in jobs:
                try:
                    name, features, feature_hash = job_name.split('__')
                    job = Job.fetch(job_id, connection = redis_con)
                
                    if job.result is None:
                        continue
                    
                    if  results_df.loc[results_df['features']==str(features), 'sharpe_ratio'].values[0] != None:
                        continue
                    
                    
                    
                    
                    test_with_states, models_used, num_models_used = job.result
                    
                    
                    backtest_results = get_backtest(test_with_states, feature_hash, features, params, models_used, num_models_used, name=name, show_plot=False)
                    

                    #print(backtest_results)
                    
                    backtest_results.to_sql('results', conn, if_exists='append')


                    sharpe_ratio = float(backtest_results['sharpe_ratio'])
                    cum_returns = float(backtest_results['cum_returns'])
                    results_df.loc[results_df['features']==str(features), 'sharpe_ratio'] = sharpe_ratio
                    results_df.loc[results_df['features']==str(features), 'cum_returns'] = cum_returns

                    if sharpe_ratio > best_sharpe_ratio:
                        if sharpe_ratio > 1:
                            plot(test_with_states, name=name, show=False)
                        best_sharpe_ratio = sharpe_ratio
                        best_features = features
                        

                except Exception as e:
                    print('EXCEPTION')
                    print(e)        
                    print(backtest_results)
                    print()            
                    sleep(5)
                    #redis_con = Redis(host='192.168.1.127')
                    #q = get_redis_connection()

            print(results_df)
            #print(len(results_df))
            print(params, results_df.head(1)['features'].values[0])
            #print('percent complete  %.2f'.format( 1-(len(results_df[results_df['sharpe_ratio'].isnull()]) / float(len(results_df))) ) )
            
            if len(results_df[results_df['sharpe_ratio'].isnull()])>2:
                #print('not complete. sleeping')
                sleep(10)
            else:
                break

            if (time.time() - start_time) > 1800: # break after 30 minutes
                print('results not found in enough time. breaking')
                break
        print('found best features', best_features, type(best_features))
        features = eval(best_features)



if __name__ == '__main__':

    
    

    n_subsets = [5,10,15]
    n_components = [4]
    lookback = [50,100,150,200]

    #features = ['return', 'rsi', 'mom']

    params = list(product( top_starting_features, n_subsets, n_components, lookback ))
    shuffle(params)

    
    p = Pool(4)
    p.map(runner, params)
    #runner(features, n_subsets, n_components, lookback)
    #runner(params[1])
    
    """                                    
    pipe_pca.fit(train.loc[:,['return', 'rsi', 'mom']])

    predictions = pipe_pca.predict(test.loc[:,['return', 'rsi', 'mom']])

    test.loc[:,'state'] = predictions

    # rename states
    new_state_df = pd.DataFrame(test.groupby(by=['state'])['return'].mean().sort_values())
    new_state_df['new_state'] = list(range(n_components))
    for i in new_state_df.index:
        test.loc[test['state']==i, 'new_state'] = new_state_df.loc[i, 'new_state']
    test['state'] = test['new_state']


    print(test.groupby(by='state')['next_return'].mean())
    print(test.groupby(by='state')['next_return'].std())
    plot(test, show=True)

    """