import numpy as np
import scipy as sc
import pandas as pd
#import TensorFlow as T
#import keras
import quandl
quandl.ApiConfig.api_key = "63gdVnc_-LzW9XyB1Ajk"

def openfiles(): 
    #cornmkt = pd.read_csv('corn/corn_JUL14.txt', header = 0)
    cornmkt = quandl.get("SCF/CME_C1_FW", authtoken="63gdVnc_-LzW9XyB1Ajk", 
        start_date="2007-05-11", end_date="2018-04-08")
    #print(cornmkt.head())
    cmd = np.array(cornmkt.index.values)
    corn_mkt_dates = []#np.chararray(cmd.shape)

    #print(corn_mkt_dates)
    for i in range(len(cmd)):
        date = (pd.to_datetime(str(cmd[i]))).strftime('%Y/%m/%d')
        #print(str(date))
        #date = date[0:4]+'/'+date[5:7]+'/'+date[8:10] 
        #corn_mkt_dates[i] = str(date)
        corn_mkt_dates.append(str(date))
    corn_mkt_dates = np.array(corn_mkt_dates)
    cornmkt.index = corn_mkt_dates
    #print(corn_mkt_dates[0])
    #cornmkt = cornmkt.drop('Date', axis=1)
    #print(cornmkt.head())
    pure_mkt_corn = np.array(cornmkt)
    

    #soymkt = pd.read_csv('soybean/soybean_JUL14.txt', header = 0)
    soymkt = quandl.get("SCF/CME_S1_FW", authtoken="63gdVnc_-LzW9XyB1Ajk", 
        start_date="2007-05-11", end_date="2018-04-08")

    smd = np.array(soymkt.index.values)
    soy_mkt_dates = []#np.chararray(smd.shape)

    #print(corn_mkt_dates)
    for i in range(len(smd)):
        date = (pd.to_datetime(str(cmd[i]))).strftime('%Y/%m/%d')
        #print(str(date))
        #date = date[0:4]+'/'+date[5:7]+'/'+date[8:10] 
        #corn_mkt_dates[i] = str(date)
        soy_mkt_dates.append(str(date))
        #date = str(smd[i])
        #date = date[0:4]+'/'+date[5:7]+'/'+date[8:10] 
        #soy_mkt_dates[i] = date
    soy_mkt_dates=np.array(soy_mkt_dates)
    soymkt.index = soy_mkt_dates
    
    #print(soymkt.head())
    pure_mkt_soy = np.array(soymkt)

    corndfs = []
    soydfs = []
    
    names = ['2007to2008', '2008to2009', '2009to2010','2010to2011', '2011to2012', '2012to2013', '2013to2014',
         '2014to2015']#, '2015to2016', '2016to2017', '2017to2018']
        
    for name in names:
        corndfs.append(pd.read_csv('corn/USDAProj_Corn_'+name+'.csv', 
            header = 0))
        soydfs.append(pd.read_csv('soybean/USDAProj_Soybean_'+name+'.csv',
            header = 0))

    fullcorn = pd.concat(corndfs, ignore_index=True, join='inner')
    fullcorn = fullcorn.truncate(after=161)
    fullcorn_dates = fullcorn['Date']
    fullcorn.index = fullcorn['Date']    
    fullcorn = fullcorn.drop('Date', axis=1)

    #fullcorn = fullcorn.truncate(before='2010/07/09')
    

    
    #print(fullcorn.head())
    
    
    
    
    fullsoy = pd.concat(soydfs, ignore_index=True, join='inner')
    fullsoy = fullsoy.truncate(after=161)
    fullsoy_dates = fullsoy['Date']

    fullsoy.index = fullsoy['Date']
    #fullsoy = fullsoy.truncate(before='2010/12/10')
    #fullsoy = fullsoy.truncate(after='2014/07/10')
    fullsoy = fullsoy.drop('Date', axis=1)
    #print(fullsoy.head(10))
    
    

    # print((cornmkt.tail()))
    # print((fullcorn.tail()))
    # print((soymkt.tail()))
    # print((fullsoy.tail()))

    # print((cornmkt.head()))
    # print((fullcorn.head()))
    # print((soymkt.head()))
    # print((fullsoy.head()))

    # print((cornmkt['Open']['2010/07/06']))

    #fullcorn = np.array(fullcorn)
    #cornmkt = np.array(cornmkt)
    #fullsoy = np.array(fullsoy)
    #soymkt = np.array(soymkt)
    return (cornmkt, soymkt, fullcorn, fullsoy, corn_mkt_dates, 
        soy_mkt_dates, np.array(fullcorn_dates), np.array(fullsoy_dates))



openfiles()

def regularize(data):
    #tranpose = data.T
    #print(data.shape)
    minv = np.min(data, axis = 0)
    maxv = np.max(data, axis = 0)
    data -= minv
    span = (maxv - minv)
    if 0.0 in span:
        for i in range(len(span)):
            if span[i] == 0:
                span[i] = 1.
    if float("inf") in span:
        for i in range(len(span)):
            if span[i] == float("inf"):
                span[i] = 1.
    data /= span
    return data


