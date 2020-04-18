import yfinance
import matplotlib
import matplotlib.pyplot as plt
from ta_indicators import get_ta
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from hmm_strategy import setup_strategy

def plot(df, name=None, show=False):
    df.loc[df['state']==0, 'color'] = 'firebrick'
    df.loc[df['state']==1, 'color'] = 'yellowgreen'
    df.loc[df['state']==2, 'color'] = 'forestgreen'
    df.loc[df['state']==3, 'color'] = 'darkslategray'

    df = df.dropna()
    df.plot.scatter(x='date',
                    y='close',
                    c='color',
                    )
                
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    if show == False and name is not None: 
        plt.savefig('./plots/%s.png' % name)
    else:
        plt.show()

    plt.close(fig)


def get_data(symbol, period):
    history = yfinance.Ticker(symbol).history(period=period, auto_adjust=False).reset_index()
    history = get_ta(history, volume=True, pattern=False)
    history.columns = map(str.lower, history.columns)
    history['return'] = history['close'].pct_change(1)    
    history['next_return'] = history['return'].shift(-1)
    history = history.dropna().reset_index(drop=True)
    return history
    

def run_feature_importances(train, n_total_features=20):
    test_cols = list(train.columns.drop(['date','open', 'high', 'low', 'close', 'return', 'next_return']))
    # get features
    clf = ExtraTreesRegressor(n_estimators=150, random_state=42)
    clf = clf.fit(train[test_cols], train['return'])
    df = pd.DataFrame([test_cols, clf.feature_importances_]).T
    df.columns = ['feature', 'importances']
    
    df = df.sort_values(by='importances')
    
    feature_choices = list(df['feature'].tail(n_total_features).values)

    preset_features = ['aroon_up', 'aroon_down', 'aroonosc','correl', 'mom', 'beta', 'rsi', 'bop', 
                        'ultimate_oscillator', 'bbands_upper', 'bbands_middle', 'bbands_lower', 
                        'bbands_upper_p', 'bbands_middle_p', 'bbands_lower_p', 'stochf_fastk', 'stochf_fastd', 'stochrsi_fastk', 'stochrsi_fastd' ]

    feature_choices = list(set( feature_choices + preset_features ))
    print('number of feature choices')
    print(len(feature_choices))
    
    top_starting_features = list(df.sort_values(by='importances').tail(10)['feature'].values)[::-1]
    return feature_choices, top_starting_features




def get_backtest(test, feature_hash, features, params, models_used, num_models_used, name=None, show_plot=False):
    import dateutil.parser

    starting_feature, n_subsets, n_components, lookback, with_rfc = params
    

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