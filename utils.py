import yfinance
import matplotlib
import matplotlib.pyplot as plt
from ta_indicators import get_ta

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
    