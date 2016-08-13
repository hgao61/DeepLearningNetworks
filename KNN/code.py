
from util import get_data, plot_data

import numpy as np
import pandas as pd
import math
import KNNLearner as knn
import BagLearner as bl
import csv
import os
import matplotlib.pyplot as plt
from pandas.tseries.offsets import *

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    
    df=port_val.copy()
    daily_ret = (df/ df.shift(1))-1
    daily_ret.fillna(0)
    cum_ret=(df[-1]-df[0])/df[0]

    
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    sharpe_ratio=((252)**0.5)*avg_daily_ret/std_daily_ret
    
    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio

def checkstockprice(date,symbol):

    orders_file = os.path.join("data", symbol+".csv")
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Open','High','Low','Close','Volume','Adj Close'], header =0)
    df=df.to_dict('records')
    
    for dicts in df:
        if dicts["Date"] == date:
            return dicts["Adj Close"]

def tradebookupdate(tradebook, symbol, numberofshares):
    if symbol in tradebook:
        
        tradebook[symbol] = tradebook[symbol] + numberofshares
    else:
        tradebook[symbol] = numberofshares
    
    return

def tradebooktotal(tradebook,date):
   sum = 0
   for (k,v) in tradebook.items():
        if k == 'CASH':
            sum = sum + v
        elif k == 'Date':
            sum = sum
        else:
            price = checkstockprice(date,k)
            subtotal = price * v
            sum = sum + subtotal
   return sum

def get_portfolio_value(prices,  start_val=1):  #updated get_portfolio_value by removing the allocs options

   
    port = prices.copy()
    for col in port:
        port[col]/=port[col].iloc[0]
    
    daily_port_val = port
    
    port_val = daily_port_val * start_val
    return port_val

def get_portfolio_tech_features(port_val, daily_rf=0, sample_per_year = 252):
    #bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    #momentum[t] = (price[t]/price[t-N]) - 1
    #volatility is just the stdev of daily returns.
    df = port_val.copy()
    sma=pd.rolling_mean(df, 20)

    std_sma = pd.rolling_std(df,20)

    daily_ret = (df/ df.shift(1))-1
    bb_value = port_val.copy()
    bb_value[0:20] =0

    bb_value[20:] = (df[20:]-sma[20:])/(20*std_sma[20:])

    momentum = (df/df.shift(-10))-1
    #momentum = momentum.shift(10)
    volatility = pd.rolling_std(daily_ret,20)#std_sma
    return bb_value, momentum, volatility

def get_portfolio_future_return(port_val, dail_rf=0, sample_per_year = 252):
    #Y[t] = (price[t+5]/price[t]) - 1.0
    df = port_val.copy()
    y5day =(df.shift(-5)/df) -1.0
    return y5day

def plot_y_data(df_bband1, df_bband2,df_bband3, filename, symbol,  title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    df1=df_bband1.copy()
    df2= df_bband2.copy()
    df3=df_bband3.copy()
    #dfevents.to_csv('plot_bollinger_band.csv')
    ax = plt.gca()

    #df.plot(ax = ax)
    #ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    df2.plot(label='Price', ax=ax,color='y', legend=True)
    df3.plot(label='Predicted Y', ax=ax,color='cyan', legend=True)
    df1.plot(label='Training Y', ax=ax,color='r',figsize=(16,10), legend=True)
    fig=plt.gcf()
    xlim = ax.get_xlim()
    factor = -0.6
    new_xlim = (xlim[0] + xlim[1])/2 + np.array((-0.5, 0.5)) * (xlim[1] - xlim[0]) * (1 + factor)
    ax.set_xlim(new_xlim)

    fig.savefig(filename)
    plt.cla()
    plt.close()

#computer_portvals is the function that execute to the order book
def updateXYs(df, savefile):
    train_port_val= df.copy()
    bb_value, momentum, volatility = get_portfolio_tech_features(train_port_val)
    #generate trainY data
    y5day = get_portfolio_future_return(train_port_val)
    #generate dataFram of trainX and trainY and save it to dataFrame
    train_tech_features = pd.concat([bb_value, momentum, volatility, y5day],  axis=1) #keys=['bb_value', 'momentum', 'volatility','y5day'],
    train_tech_features.columns=['bb_value', 'momentum', 'volatility','y5day']
    train_tech_features.fillna(0,inplace=True)
    #save norminized test data to cvs and
    # calculate tech features out of it

    train_tech_features.to_csv(savefile,index=False,header=False)
    #Open cvs file and assign to trainX and trainY
    inf = open(savefile)
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    trainX = data[0:data.shape[0],0:-1]
    trainY = data[0:data.shape[0],-1]
    return trainX, trainY

def scale_data_with_train_prices(df):
    train_prices = df.copy()
    y5day_scaled=train_prices.shift(-5)
    y5day_scaled.fillna(0,inplace=True)
    y5day_scaled.columns =['Training Y']
    predYpd_scaled= train_prices.ix[:,0] + train_prices.ix[:,0] * predY
    df_predYpd_scaled = pd.DataFrame(predYpd_scaled, index=y5day_scaled.index)
    df_predYpd_scaled.columns =['Predicted Y']
    #print 'predYpd_scaled',predYpd_scaled
    train_prices.columns =['Price']
    return y5day_scaled, train_prices,df_predYpd_scaled

def listworkingdays(symbol):
    orders_file = os.path.join("data", symbol+".csv")
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Open','High','Low','Close','Volume','Adj Close'], header =0)
    df=df.to_dict('list')
    return df["Date"]

def keyword_create(row):
    if row['delta'] >0.005:
        return 'longentry'
    return 0

def trading_events_generator(test_prices_test, df_predYpd_scaled_test):
    dfP = test_prices_test.copy()
    dfY = df_predYpd_scaled_test.copy()
    dfY.to_csv('dfY.csv')

    dfY['delta'] =((dfY['Predicted Y']-dfY['Predicted Y'].shift(5))/dfY['Predicted Y']).fillna(0)
    dfY['delta'] = dfY['delta'].shift(-5)
    #newdfY=dfY[dfY.delta>0.1]
    dfY['events']= dfY.apply(lambda row: keyword_create(row), axis =1)
    #newdfY['events']= 'longentry'
    events= dfY
    events = events.drop('Predicted Y', 1)
    events = events.drop('delta', 1)
    return events

def compute_portvals(start_date, end_date, orders_file, start_val):

    #print 'orders_file',orders_file
    df = pd.read_csv(orders_file, delimiter=r",", names=['Date','Shares','Symbol','Order'], header =0)
    dfrecords=df.to_dict('records')
    initial_stocklist=[{'Date': start_date, 'CASH':start_val}]
    stocklist = initial_stocklist
    listofdates= listworkingdays('SPY')
    workingdates =[]
    for dates in pd.date_range(start=start_date, end=end_date, freq='D'):
        #print dates
        if dates.strftime('%Y-%m-%d') in listofdates:
            workingdates.append(dates.strftime('%Y-%m-%d'))
    #print workingdates
    total=[]
    newentry={'Date': start_date, 'CASH':start_val}
    for i in workingdates:
        today = i
        yesterday = pd.to_datetime(i) + pd.DateOffset(days=-1)
        tomorrow = pd.to_datetime(i) + pd.DateOffset(days=1)
        for dicts1 in stocklist:
            #print dicts1
            if dicts1["Date"] == today:
                newentry = dicts1.copy()

        for dicts in dfrecords:
            if dicts["Date"] ==i:
                if dicts["Order"] == "BUY":
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   newentry["CASH"] = newentry["CASH"] - stockprice * dicts["Shares"]
                   tradebookupdate(newentry,dicts["Symbol"],dicts["Shares"])
                elif dicts["Order"]  == "SELL":
                   quantity = -dicts["Shares"]
                   stockprice=checkstockprice(today, dicts["Symbol"])
                   newentry["CASH"] = newentry["CASH"] + stockprice * dicts["Shares"]
                   tradebookupdate(newentry,dicts["Symbol"],quantity)
            #else:
                #newentry["CASH"] = newentry["CASH"]
        newentry["Date"] = tomorrow.strftime('%Y-%m-%d')
        total.append(tradebooktotal(newentry, today))
        stocklist.append(newentry.copy())

    portvals = pd.DataFrame(total, index=workingdates, columns=['Portfolio'])
    return portvals

def plot_events_data(test_prices_test, df_predYpd_scaled_test, filename, symbol, df_events, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    dfP=test_prices_test.copy()
    dfY= df_predYpd_scaled_test.copy()
    dfevents=df_events
    #dfevents.to_csv('plot_bollinger_band.csv')
    ax = plt.gca()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    dfP.plot(label=symbol, ax=ax,color='b',figsize=(16,10), legend=True)
    dfY.plot(label='Predicated Y', ax=ax,color='y',figsize=(16,10), legend=True)

    longentry = dfevents[dfevents['events'] =='longentry']
    longexit = dfevents[dfevents['events'] =='longexit']
    shortentry = dfevents[dfevents['events'] =='shortentry']
    shortexit = dfevents[dfevents['events'] =='shortexit']
    #print longentry.index
    plt.vlines(longentry.index, plt.ylim()[0], plt.ylim()[1] ,color ='g',linewidth=1)
    plt.vlines(longexit.index, plt.ylim()[0], plt.ylim()[1],color ='k')
    plt.vlines(shortentry.index, plt.ylim()[0], plt.ylim()[1],color ='r')
    plt.vlines(shortexit.index, plt.ylim()[0], plt.ylim()[1],color ='k')

    fig=plt.gcf()
    xlim = ax.get_xlim()
    factor = -0.6

    new_xlim = (xlim[0] + xlim[1])/2 + np.array((-0.5, 0.5)) * (xlim[1] - xlim[0]) * (1 + factor)
    #print new_xlim
    ax.set_xlim(new_xlim)
    fig.savefig(filename)
    plt.close()

def orderbook_generator(events, symbol, stock_shares=100 ):
    df= events.copy()
    #print df[df['events'].isin(['longentry', 'longexit','shortentry','shortexit'])]
    columns = ['Date','Shares', 'Symbol', 'Order']
    order_book = pd.DataFrame(data=np.zeros((0,len(columns))), columns=columns)
    columns2 = ['events',]
    updated_events = pd.DataFrame(data=np.zeros((0,len(columns2))), columns=columns2)
    #print 'order_book', order_book
    flag_long =0
    flag_short =0
    counter = 5
    for i in range(0,len(df.index)):

        if (df['events'][i] =='longentry') and (flag_long ==0) and (counter >0):
            flag_long = 1
            counter = counter -1
            order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'BUY','Shares':stock_shares},ignore_index=True)
            updated_events = updated_events.append({'Date': df.index[i], 'events':'longentry'},ignore_index=True)
        elif (flag_long ==1) and (counter >0):
            counter = counter -1
        elif (flag_long ==1) and (counter ==0):
            counter =5
            flag_long =0
            order_book = order_book.append({'Date':df.index[i], 'Symbol':symbol, 'Order':'SELL','Shares':stock_shares},ignore_index=True)
            updated_events = updated_events.append({'Date': df.index[i], 'events':'longexit'},ignore_index=True)

    order_book1 = order_book.set_index(['Date'])
    #print 'order_book', order_book1
    order_book1.to_csv('orders.csv')
    #print 'updated_events', updated_events
    updated_events1 = updated_events.set_index(['Date'])
    #updated_events1.to_csv('events.csv')
    return updated_events1

if __name__=="__main__":

    # compute how much of the data is training and testing
    train_start_date = '2008-01-01'
    train_end_date = '2009-12-31'
    test_start_date = '2010-01-01'
    test_end_date = '2010-12-31'
    symbols =['IBM']

     #symbols =['ML4T-399']
    number_of_stocks = 100
    start_val = 10000
    train_dates= pd.date_range(train_start_date, train_end_date)
    test_dates = pd.date_range(test_start_date, test_end_date)
    #
    train_prices_all = get_data(symbols,train_dates)
    test_prices_all = get_data(symbols,test_dates)

    train_prices = train_prices_all[symbols]
    test_prices = test_prices_all[symbols]
    train_port_val = get_portfolio_value(train_prices) #norminized train_prices
    test_port_val = get_portfolio_value(test_prices) #norminized test prices

    trainX, trainY = updateXYs(train_port_val,'save_trainXY.csv')

    testX, testY = updateXYs(test_port_val,'save_testXY.csv')

    # create a learner and train it
    #learner = bl.BagLearner(learner =  knn.KNNLearner, kwargs={"k":3}, bags= 100, boost = False)
    learner = knn.KNNLearner(k=3) # create a KNNLeaner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    ############################
    y5day_scaled, train_prices,df_predYpd_scaled = scale_data_with_train_prices(train_prices)
    #print y5day_scaled
    plot_y_data(y5day_scaled[:-5],train_prices[:-5], df_predYpd_scaled[:-5],'data_trY_price_predY.png',symbols, title="Stock prices", xlabel="Date", ylabel="Price")
    events_train = trading_events_generator(test_prices, df_predYpd_scaled.shift(5))
    updated_events_train = orderbook_generator(events_train, ''.join(symbols), number_of_stocks)
    plot_events_data(test_prices,df_predYpd_scaled,'data_insample.png',symbols, updated_events_train, title="Predicted Y")
    ############################

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
    ############################
    y5day_scaled_test, test_prices_test,df_predYpd_scaled_test = scale_data_with_train_prices(test_prices)
    events = trading_events_generator(test_prices_test, df_predYpd_scaled_test.shift(5))
    updated_events = orderbook_generator(events, ''.join(symbols), number_of_stocks)
    plot_events_data(test_prices_test,df_predYpd_scaled_test,'data_outofsample.png',symbols, updated_events, title="Predicted Y")
    ############################
    portvals = compute_portvals(test_start_date, test_end_date, "orders.csv", start_val)

    #plot_normalized_data(portvals, 'Daily portfolio value.png', title="Daily portfolio value")

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    #print 'portvals in test_run', portvals
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(test_start_date, test_end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
