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
import io
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




def get_backtest(test, show_plot=False):
    import dateutil.parser

    symbols = ['SPY', 'SSO', 'UPRO']
    files = []
    for symbol in symbols:
        filename = './datasets/%s.csv' % symbol
        try:
            df = pd.read_csv(filename)
            
        except:
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
        
    backtest_results = setup_strategy(files, name, show_plot=show_plot)
    print(backtest_results)
    print(type(backtest_results))
    backtest_results['avg_per_day'] = backtest_results.loc['cum_returns'] / float(len(test))
    backtest_results['name'] = name
    backtest_results['features'] = str(features)
    backtest_results['num_features'] = len(features)
    backtest_results['n_subsets'] = n_subsets
    backtest_results['n_components'] = n_components
    backtest_results['lookback'] = lookback

    return backtest_results
    
if __name__ == '__main__':

    history_df = get_data('SPY', period='3y')

    train = history_df.loc[history_df['date']<'2019-01-01']
    test = history_df.loc[history_df['date']>'2019-01-01']

    train.to_csv('./datasets/train.csv', index=False)
    test.to_csv('./datasets/test.csv', index=False)
    
    n_subsets = 10
    n_components = 4
    lookback = 100
    features = ['return', 'rsi', 'mom']
    name = namegenerator.gen()

    test_with_states, models_used, num_models_used = generate_model(features, n_subsets, n_components, lookback, name)

    backtest_results = get_backtest(test_with_states, show_plot=False)
    states_plot = plot(test_with_states, name=name, show=False)
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