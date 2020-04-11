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


def run_decision_tree(train):
    test_cols = list(train.columns.drop(['date','open', 'high', 'low', 'close', 'return', 'next_return']))
    # get features
    clf = ExtraTreesRegressor(n_estimators=150)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances')
    
    feature_choices = df['feature'].tail(40).values
    top_starting_features = list(df.sort_values(by='importances').tail(10)['feature'].values)
    return feature_choices, top_starting_features


history_df = get_data('SPY', period='3y')

train = history_df.loc[history_df['date']<'2019-01-01']
test = history_df.loc[history_df['date']>'2019-01-01']

train.to_csv('./datasets/train.csv', index=False)
test.to_csv('./datasets/test.csv', index=False)

feature_choices, top_starting_features = run_decision_tree(train)
top_starting_features = ['intraday_change']


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




def get_backtest(test, features, params, models_used, num_models_used, name=None, show_plot=False):
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
    backtest_results['features'] = starting_feature
    backtest_results['features'] = str(features)
    backtest_results['num_features'] = len(features)
    
    backtest_results['n_subsets'] = n_subsets
    backtest_results['n_components'] = n_components
    backtest_results['lookback'] = lookback

    return backtest_results


def runner(params):
    print('starting process!')
    conn =  sqlite3.connect('results.db')
    print(params)
    starting_feature, n_subsets, n_components, lookback = params
    shuffle(feature_choices)
    features = [starting_feature]
    while len(features)<16:
        q = Queue(connection=Redis( host='192.168.1.127' ))
        jobs = []
        for new_feature in feature_choices:

            test_features = features + [new_feature]
            
            print(test_features)
            
            
            name = namegenerator.gen()

            job_id = name+'__'+str(test_features)

            job = q.enqueue(generate_model, args = (test_features, n_subsets, n_components, lookback, name, ), job_timeout=3600,  result_ttl=86400 )
            jobs.append( (job, job_id) )

        
        start_time = time.time()
        
        results_df = pd.DataFrame()
        
        while True:
            
            for job, job_id in jobs:
                features = job_id.split('__')[1]
                if job.result is None:
                    sharpe_ratio = None
                else:    
                    """
                    sharpe_ratio = job.result[4]
                    if sharpe_ratio > best_sharpe_ratio:
                        best_sharpe_ratio = sharpe_ratio
                        best_features = features
                        best_job_results = [job.result[1], job.result[2], job.result[3]]
                    """
                    test_with_states, models_used, num_models_used = job.result
                    backtest_results = get_backtest(test_with_states, test_features, params, models_used, num_models_used, name=name, show_plot=False)
                    print(backtest_results)
                    sharpe_ratio = float(backtest_results['sharpe_ratio'])
                    results_df.loc[features, sharpe_ratio]


                #results_df.append( [name, features, sharpe_ratio] )
            print(results_df)
            sleep(5)            
            #results_df = pd.DataFrame(results_df, columns = ['name', 'features', 'sharpe_ratio'])

            #print(results_df)
            #print(  len(results_df.dropna()) / float(len(results_df)) )
            
            """
            if len(results_df[results_df['sharpe_ratio'].isnull()]):
                print('not complete. sleeping')
                sleep(5)
            else:
                break

            if (time.time() - start_time) > 1800: # break after thirty minutes
                print('results not found in enough time. breaking')
                break

            #test_with_states, models_used, num_models_used = generate_model(test_features, n_subsets, n_components, lookback, name)
            print(test_with_states)
            backtest_results = get_backtest(test_with_states, test_features, params, models_used, num_models_used, name=name, show_plot=False)
            plot(test_with_states, name=name, show=False)
            print(test_features)
            print(backtest_results)
            
            backtest_results.to_sql('results', conn, if_exists='append')
            """





if __name__ == '__main__':

    
    

    n_subsets = [5,10,15]
    n_components = [4]
    lookback = [50,100,150,200]

    #features = ['return', 'rsi', 'mom']

    params = list(product( top_starting_features, n_subsets, n_components, lookback ))
    shuffle(params)

    
    #p = Pool(1)
    #p.map(runner, params)
    #runner(features, n_subsets, n_components, lookback)
    runner(params[1])
    
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