from __future__ import print_function

from pyalgotrade import strategy
from pyalgotrade.barfeed import quandlfeed, yahoofeed, csvfeed,googlefeed

from pyalgotrade.technical import ma
import pandas as pd

from pyalgotrade import dataseries
from pyalgotrade import technical
#from pyalgotrade.order import BUY, SELL
from math import floor
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades
from pyalgotrade.bar import Frequency
from pyalgotrade import plotter
import matplotlib
import numpy as np
import yfinance
import os

class AccuracyStrat(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument_1, instrument_2, instrument_3, states_instrument, smaPeriod=1):
        super(AccuracyStrat, self).__init__(feed, 20000)
        self.__position = None
        self.__instrument_1 = instrument_1
        self.__instrument_2 = instrument_2
        self.__instrument_3 = instrument_3

        self.instruments = [self.__instrument_1, self.__instrument_2, self.__instrument_3]
        
        self.states_instrument = states_instrument
        #print('got states instrument', states_instrument)
        #print(list(feed['SPY'].getCloseDataSeries()))
        #self.__my_indicator = ma.SMA(feed[states_instrument].getExtraDataSeries('State'), smaPeriod)
        self.__my_indicator = ma.SMA(feed[states_instrument].getCloseDataSeries(), smaPeriod)

        self.usage = {}
        
    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    def onBars(self, bars):
        
        
        state = self.__my_indicator[-1]
        
        #print(state)
        #input()
        if state is None:
            return

        self.usage[self.__instrument_1] = 0
        self.usage[self.__instrument_2] = 0
        self.usage[self.__instrument_3] = 0

        if state == 1:
            self.usage[self.__instrument_1] = .95
            self.usage[self.__instrument_2] = 0
            self.usage[self.__instrument_3] = 0

        elif state == 2:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = .95
            self.usage[self.__instrument_3] = 0

        elif state == 3 and self.__instrument_3 is not None:
            self.usage[self.__instrument_1] = 0
            self.usage[self.__instrument_2] = 0
            self.usage[self.__instrument_3] = .95

        #bar = bars.getBar('QQQ')
        #print('bar', bar)
        for instrument in self.instruments:
            if instrument is None:
                continue
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*.9) )

            currentPos = self.getBroker().getShares(instrument)

            num_shares = int(num_shares - currentPos)

            #if num_shares<-500:
                #num_shares = -500

            if num_shares<0:
                #self.limitOrder(instrument, close * 0.9, num_shares)
                #print('sell order', instrument, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)
        
        for instrument in self.instruments:
            if instrument is None:
                continue
            bar = bars.getBar(instrument)
            close = bar.getClose()
            
            usage = self.usage[instrument]

            num_shares = floor( (self.getBroker().getEquity() * usage)  / (close*1.1) )
            
            currentPos = self.getBroker().getShares(instrument)
            
            num_shares = int(num_shares - currentPos)
            #if num_shares>500:
                #num_shares = 500
            if num_shares>0:
                #self.limitOrder(instrument, close * 1.1, num_shares)
                #print('buy order', instrument, num_shares)
                self.marketOrder(instrument, num_shares, onClose=True)


def setup_strategy(files, symbols, name, show_plot=False):
    #from pyalgotrade.feed import csvfeed, yahoofeed

    # Load the bar feed from the CSV file
    #feed = csvfeed.GenericBarFeed(frequency=Frequency.DAY)
    feed = yahoofeed.Feed(Frequency.DAY)
    #feed = csvfeed.Feed("Date", "%Y-%m-%d")

    for sym, filename in files:
        #print('loading', sym, filename)
        if 'predictions' not in filename:
            feed.addBarsFromCSV(sym, filename)
        else:
            feed.addBarsFromCSV('states', filename)
            
    
    
    
    instrument_1 = files[0][0]
    instrument_2 = files[1][0]
    if len(files)==3:
        instrument_3 = files[2][0]
    else:
        instrument_3 = None
    states_instrument = 'states'
    
    #print('got these instruments', instrument_1, instrument_2, instrument_3, states_instrument)
    #input()
    #print(files)
    # Evaluate the strategy with the feed.
    myStrategy = AccuracyStrat(feed, instrument_1, instrument_2, instrument_3, states_instrument)

    
    from pyalgotrade.stratanalyzer import returns
    # Attach different analyzers to a strategy before executing it.
    retAnalyzer = returns.Returns()
    myStrategy.attachAnalyzer(retAnalyzer)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    myStrategy.attachAnalyzer(sharpeRatioAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    myStrategy.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    myStrategy.attachAnalyzer(tradesAnalyzer)

    # Attach the plotter to the strategy.
    plt = plotter.StrategyPlotter(myStrategy)
    
    
    # Include the SMA in the instrument's subplot to get it displayed along with the closing prices.
    #plt.getInstrumentSubplot("orcl").addDataSeries("SMA", myStrategy.getSMA())
    # Plot the simple returns on each bar.
    plt.getOrCreateSubplot("returns").addDataSeries("Simple returns", retAnalyzer.getReturns())
    

    
    #

    # Run the strategy.
    myStrategy.run()

    
    if show_plot:
        plt.plot()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 5)
        
    cum_returns = retAnalyzer.getCumulativeReturns()[-1] * 100
    sharpe_ratio = sharpeRatioAnalyzer.getSharpeRatio(0.05)
    results = {}
    results['final_value'] = myStrategy.getResult()
    results['cum_returns'] = cum_returns
    results['sharpe_ratio'] = sharpe_ratio
    if not show_plot and sharpe_ratio>1 and cum_returns>200:
        plt.savePlot('./backtest_plots/%s_%s.png' % ( str(int(results['cum_returns'])), name ))
    
    
    del plt
    
    results['max_drawdown_%'] = drawDownAnalyzer.getMaxDrawDown() * 100
    results['longest_drawdown'] = str(drawDownAnalyzer.getLongestDrawDownDuration())

    results['total_trades'] = tradesAnalyzer.getCount()
    results['profitable_trades'] = tradesAnalyzer.getProfitableCount()

    try:
        results['win_rate'] = tradesAnalyzer.getProfitableCount() / tradesAnalyzer.getCount()
    

        profits = tradesAnalyzer.getAll()
        results['avg_profit_$'] = profits.mean()
        results['std_profit_$'] = profits.std()
        results['max_profit_$'] = profits.max()
        results['min_profit_$'] = profits.min()
        
        returns = tradesAnalyzer.getAllReturns()
        results['avg_profit_%'] = returns.mean() * 100
        results['std_profit_%'] = returns.std() * 100
        results['max_profit_%'] = returns.max() * 100
        results['min_profit_%'] = returns.min() * 100

        for sym, filename in files:
            os.remove(filename)

        
    except Exception as e:
        #print('backtest exception', e)
        pass
    results = pd.DataFrame.from_dict(results, orient='index')
    #print(results)

    return results

if __name__ == "__main__":
    
    print("Final portfolio value: $%.2f" % myStrategy.getResult())
    print("Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
    print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
    print("Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100))
    print("Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))

    print("")
    print("Total trades: %d" % (tradesAnalyzer.getCount()))
    if tradesAnalyzer.getCount() > 0:
        profits = tradesAnalyzer.getAll()
        print("Avg. profit: $%2.f" % (profits.mean()))
        print("Profits std. dev.: $%2.f" % (profits.std()))
        print("Max. profit: $%2.f" % (profits.max()))
        print("Min. profit: $%2.f" % (profits.min()))
        returns = tradesAnalyzer.getAllReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

    print("")
    print("Profitable trades: %d" % (tradesAnalyzer.getProfitableCount()))
    if tradesAnalyzer.getProfitableCount() > 0:
        profits = tradesAnalyzer.getProfits()
        print("Avg. profit: $%2.f" % (profits.mean()))
        print("Profits std. dev.: $%2.f" % (profits.std()))
        print("Max. profit: $%2.f" % (profits.max()))
        print("Min. profit: $%2.f" % (profits.min()))
        returns = tradesAnalyzer.getPositiveReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

    print("")
    print("Unprofitable trades: %d" % (tradesAnalyzer.getUnprofitableCount()))
    if tradesAnalyzer.getUnprofitableCount() > 0:
        losses = tradesAnalyzer.getLosses()
        print("Avg. loss: $%2.f" % (losses.mean()))
        print("Losses std. dev.: $%2.f" % (losses.std()))
        print("Max. loss: $%2.f" % (losses.min()))
        print("Min. loss: $%2.f" % (losses.max()))
        returns = tradesAnalyzer.getNegativeReturns()
        print("Avg. return: %2.f %%" % (returns.mean() * 100))
        print("Returns std. dev.: %2.f %%" % (returns.std() * 100))
        print("Max. return: %2.f %%" % (returns.max() * 100))
        print("Min. return: %2.f %%" % (returns.min() * 100))

    