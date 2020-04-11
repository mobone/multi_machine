from talib import *
from talib._ta_lib import *
from more_indicators import *
import pandas as pd
#[a-zA-Z, ]* \= [a-zA-Z0-9_]*\([a-zA-Z, _=0-9]*\)



def get_ta(df, volume, pattern):
    if volume is None:
        open, high, low, close = df['Open'], df['High'], df['Low'], df['Close']
    else:
        open, high, low, close, volume = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']

    if 'Dividends' in df.columns:
        del df['Dividends']
    if 'Stock Splits' in df.columns:
        del df['Stock Splits']

    df['MACD'],df['MACD_signal'],df['MACD_hist'] = MACD(close)
    df['RSI'] = RSI(close)

    df['intraday_change'] = df['Close'] / df['Open'] - 1
    
    #for i in range(1,2):
    #    df['%s_day_change' % i] = df['Close'].shift(i) / df['Close'] - 1

    """
    MA
    EMA
    BOLL    
    SAR
    Pivot point
    DC
    VOL
    MACD
    RSI
    ROC
    Aroon
    ADX
    MFI
    ATR
    CCI
    W%R
    Net Volume
    OBV
    Percent B
    Chaikin Oscillator


    ADL*
    EFI*
    CC*
    DPO*
    UOS*
    DMI*
    HA*
    DMA*
    FSTO*
    KC*
    IC*
    VWAP*
    KDJ*
    """
    df = PPSR(df)
    df['MA'] = MA(close)
    
    df['EMA'] = EMA(close)
    df['BBANDS_upper'],df['BBANDS_middle'],df['BBANDS_lower'] = BBANDS(close)
    df['SAR'] = SAR(high, low)
    df['HT_DCPERIOD'] = HT_DCPERIOD(close)
    df['HT_DCPHASE'] = HT_DCPHASE(close)
    df['HT_PHASOR_inphase'],df['HT_PHASOR_quadrature'] = HT_PHASOR(close)

    
    df['ROC'] = ROC(close)
    df['MOM'] = MOM(close)
    df['DX'] = DX(high, low, close)
    df['AROON_down'],df['AROON_up'] = AROON(high, low)
    df['AROONOSC'] = AROONOSC(high, low)
    df['ADX'] = ADX(high, low, close)
    if volume is not None:
        df['MFI'] = MFI(high, low, close, volume)
        df['OBV'] = OBV(close, volume)
    df['ATR'] = ATR(high, low, close)
    df['CCI'] = CCI(high, low, close)
    df['BETA'] = BETA(high, low)
    df['CORREL'] = CORREL(high, low)
    df['TRANGE'] = TRANGE(high, low, close)
    df['STOCH_slowk'],df['STOCH_slowk'] = STOCH(high, low, close)
    df['STOCHF_fastk'],df['STOCHF_fastd'] = STOCHF(high, low, close)
    df['STOCHRSI_fastk'],df['STOCHRSI_fastd'] = STOCHRSI(close)
    df['HT_TRENDLINE'] = HT_TRENDLINE(close)

    # more indicators
    df = williams_r(df, high_col = 'High', low_col = 'Low', close_col = 'Close')
    df = chaikin_oscillator(df, high_col = 'High', low_col = 'Low', close_col = 'Close',  vol_col = 'Volume')
    df = chaikin_volatility(df, high_col = 'High', low_col = 'Low', close_col = 'Close')
    df = ultimate_oscillator(df, high_col = 'High', low_col = 'Low', close_col = 'Close')
    df = price_volume_trend(df, close_col = 'Close',  vol_col = 'Volume')
    df = negative_volume_index(df, close_col = 'Close',  vol_col = 'Volume')
    df = positive_volume_index(df, close_col = 'Close',  vol_col = 'Volume')


    for col in ['BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower']:
            df[col+'_p'] = df['Close'] / df[col] - 1
    
    df['WMA'] = WMA(close)
    
    
    df['DEMA'] = DEMA(close)
    
    df['HT_TRENDLINE'] = HT_TRENDLINE(close)
    df['KAMA'] = KAMA(close)
    
    df['MAMA'],df['FAMA'] = MAMA(close)
    #df['MAVP'] = MAVP(close, periods)
    df['MIDPOINT'] = MIDPOINT(close)
    df['MIDPRICE'] = MIDPRICE(high, low)
    
    df['SAREXT'] = SAREXT(high, low)
    df['SMA'] = SMA(close)
    df['T3'] = T3(close)
    df['TEMA'] = TEMA(close)
    df['TRIMA'] = TRIMA(close)
    df['WMA'] = WMA(close)

    df['ADX'] = ADX(high, low, close)
    df['ADXR'] = ADXR(high, low, close)
    df['APO'] = APO(close)
    df['AROON_down'],df['AROON_up'] = AROON(high, low)
    df['AROONOSC'] = AROONOSC(high, low)
    df['BOP'] = BOP(open, high, low, close)
    df['CCI'] = CCI(high, low, close)
    df['CMO'] = CMO(close)
    df['DX'] = DX(high, low, close)
    df['MACD'],df['MACD_signal'],df['MACD_hist'] = MACD(close)
    df['MACDFIX'],df['MACDFIX_signal'],df['MACDFIX_hist'] = MACDFIX(close)
    
    df['MINUS_DI'] = MINUS_DI(high, low, close)
    df['MINUS_DM'] = MINUS_DM(high, low)
    df['MOM'] = MOM(close)
    df['PLUS_DI'] = PLUS_DI(high, low, close)
    df['PLUS_DM'] = PLUS_DM(high, low)
    df['PPO'] = PPO(close)
    df['ROC'] = ROC(close)
    df['ROCP'] = ROCP(close)
    df['ROCR'] = ROCR(close)
    df['ROCR100'] = ROCR100(close)
    df['RSI'] = RSI(close)
    df['STOCH_slowk'],df['STOCH_slowk'] = STOCH(high, low, close)
    df['STOCHF_fastk'],df['STOCHF_fastd'] = STOCHF(high, low, close)
    df['STOCHRSI_fastk'],df['STOCHRSI_fastd'] = STOCHRSI(close)
    df['TRIX'] = TRIX(close)
    df['ULTOSC'] = ULTOSC(high, low, close)
    df['WILLR'] = WILLR(high, low, close)
    if volume is not None:
        df['AD'] = AD(high, low, close, volume)
        df['ADOSC'] = ADOSC(high, low, close, volume)
        df['OBV'] = OBV(close, volume)

    df['HT_DCPERIOD'] = HT_DCPERIOD(close)
    df['HT_DCPHASE'] = HT_DCPHASE(close)
    df['HT_PHASOR_inphase'],df['HT_PHASOR_quadrature'] = HT_PHASOR(close)
    df['HT_SINE'],df['HT_LEADSINE'] = HT_SINE(close)
    df['HT_TRENDMODE'] = HT_TRENDMODE(close)

    df['ATR'] = ATR(high, low, close)
    df['NATR'] = NATR(high, low, close)
    df['TRANGE'] = TRANGE(high, low, close)
    if pattern:
        
        df['CDL2CROWS'] = CDL2CROWS(open, high, low, close)
        df['CDL3BLACKCROWS'] = CDL3BLACKCROWS(open, high, low, close)
        df['CDL3INSIDE'] = CDL3INSIDE(open, high, low, close)
        df['CDL3LINESTRIKE'] = CDL3LINESTRIKE(open, high, low, close)
        df['CDL3OUTSIDE'] = CDL3OUTSIDE(open, high, low, close)
        df['CDL3STARSINSOUTH'] = CDL3STARSINSOUTH(open, high, low, close)
        df['CDL3WHITESOLDIERS'] = CDL3WHITESOLDIERS(open, high, low, close)
        df['CDLABANDONEDBABY'] = CDLABANDONEDBABY(open, high, low, close)
        df['CDLADVANCEBLOCK'] = CDLADVANCEBLOCK(open, high, low, close)
        df['CDLBELTHOLD'] = CDLBELTHOLD(open, high, low, close)
        df['CDLBREAKAWAY'] = CDLBREAKAWAY(open, high, low, close)
        df['CDLCLOSINGMARUBOZU'] = CDLCLOSINGMARUBOZU(open, high, low, close)
        df['CDLCONCEALBABYSWALL'] = CDLCONCEALBABYSWALL(open, high, low, close)
        df['CDLCOUNTERATTACK'] = CDLCOUNTERATTACK(open, high, low, close)
        df['CDLDARKCLOUDCOVER'] = CDLDARKCLOUDCOVER(open, high, low, close)
        df['CDLDOJI'] = CDLDOJI(open, high, low, close)
        df['CDLDOJISTAR'] = CDLDOJISTAR(open, high, low, close)
        df['CDLDRAGONFLYDOJI'] = CDLDRAGONFLYDOJI(open, high, low, close)
        df['CDLENGULFING'] = CDLENGULFING(open, high, low, close)
        df['CDLEVENINGDOJISTAR'] = CDLEVENINGDOJISTAR(open, high, low, close)
        df['CDLEVENINGSTAR'] = CDLEVENINGSTAR(open, high, low, close)
        df['CDLGAPSIDESIDEWHITE'] = CDLGAPSIDESIDEWHITE(open, high, low, close)
        df['CDLGRAVESTONEDOJI'] = CDLGRAVESTONEDOJI(open, high, low, close)
        df['CDLHAMMER'] = CDLHAMMER(open, high, low, close)
        df['CDLHANGINGMAN'] = CDLHANGINGMAN(open, high, low, close)
        df['CDLHARAMI'] = CDLHARAMI(open, high, low, close)
        df['CDLHARAMICROSS'] = CDLHARAMICROSS(open, high, low, close)
        df['CDLHIGHWAVE'] = CDLHIGHWAVE(open, high, low, close)
        df['CDLHIKKAKE'] = CDLHIKKAKE(open, high, low, close)
        df['CDLHIKKAKEMOD'] = CDLHIKKAKEMOD(open, high, low, close)
        df['CDLHOMINGPIGEON'] = CDLHOMINGPIGEON(open, high, low, close)
        df['CDLIDENTICAL3CROWS'] = CDLIDENTICAL3CROWS(open, high, low, close)
        df['CDLINNECK'] = CDLINNECK(open, high, low, close)
        df['CDLINVERTEDHAMMER'] = CDLINVERTEDHAMMER(open, high, low, close)
        df['CDLKICKING'] = CDLKICKING(open, high, low, close)
        df['CDLKICKINGBYLENGTH'] = CDLKICKINGBYLENGTH(open, high, low, close)
        df['CDLLADDERBOTTOM'] = CDLLADDERBOTTOM(open, high, low, close)
        df['CDLLONGLEGGEDDOJI'] = CDLLONGLEGGEDDOJI(open, high, low, close)
        df['CDLLONGLINE'] = CDLLONGLINE(open, high, low, close)
        df['CDLMARUBOZU'] = CDLMARUBOZU(open, high, low, close)
        df['CDLMATCHINGLOW'] = CDLMATCHINGLOW(open, high, low, close)
        df['CDLMATHOLD'] = CDLMATHOLD(open, high, low, close)
        df['CDLMORNINGDOJISTAR'] = CDLMORNINGDOJISTAR(open, high, low, close)
        df['CDLMORNINGSTAR'] = CDLMORNINGSTAR(open, high, low, close)
        df['CDLONNECK'] = CDLONNECK(open, high, low, close)
        df['CDLPIERCING'] = CDLPIERCING(open, high, low, close)
        df['CDLRICKSHAWMAN'] = CDLRICKSHAWMAN(open, high, low, close)
        df['CDLRISEFALL3METHODS'] = CDLRISEFALL3METHODS(open, high, low, close)
        df['CDLSEPARATINGLINES'] = CDLSEPARATINGLINES(open, high, low, close)
        df['CDLSHOOTINGSTAR'] = CDLSHOOTINGSTAR(open, high, low, close)
        df['CDLSHORTLINE'] = CDLSHORTLINE(open, high, low, close)
        df['CDLSPINNINGTOP'] = CDLSPINNINGTOP(open, high, low, close)
        df['CDLSTALLEDPATTERN'] = CDLSTALLEDPATTERN(open, high, low, close)
        df['CDLSTICKSANDWICH'] = CDLSTICKSANDWICH(open, high, low, close)
        df['CDLTAKURI'] = CDLTAKURI(open, high, low, close)
        df['CDLTASUKIGAP'] = CDLTASUKIGAP(open, high, low, close)
        df['CDLTHRUSTING'] = CDLTHRUSTING(open, high, low, close)
        df['CDLTRISTAR'] = CDLTRISTAR(open, high, low, close)
        df['CDLUNIQUE3RIVER'] = CDLUNIQUE3RIVER(open, high, low, close)
        df['CDLUPSIDEGAP2CROWS'] = CDLUPSIDEGAP2CROWS(open, high, low, close)
        df['CDLXSIDEGAP3METHODS'] = CDLXSIDEGAP3METHODS(open, high, low, close)
    

    df['BETA'] = BETA(high, low)
    df['CORREL'] = CORREL(high, low)
    df['LINEARREG'] = LINEARREG(close)
    df['LINEARREG_ANGLE'] = LINEARREG_ANGLE(close)
    df['LINEARREG_INTERCEPT'] = LINEARREG_INTERCEPT(close)
    df['LINEARREG_SLOPE'] = LINEARREG_SLOPE(close)
    df['STDDEV'] = STDDEV(close)
    df['TSF'] = TSF(close)
    df['VAR'] = VAR(close)
    
    """
    for col in df.columns.drop(['Adj Close', 'Volume', 'Close', 'Open', 'High', 'Low']):

        #percent_difference = ( df['Close']/df[col] ).tail(100).mean()
        correl = df[[col,'Close']].corr()
        
        this_correl = correl[col]['Close']
        
        if abs(this_correl)>.90:
        

            df['diff_'+col] = df['Close'] - df[col]
            
            df['diff_p_'+col] = df['Close'] / df[col] - 1
    """
    #print('after')
    #print(df)
    return df
