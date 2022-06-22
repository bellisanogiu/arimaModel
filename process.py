#deleted 21/2/2012 from WBA.csv and from TPR.csv
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime

start_date = sys.argv[1]
days_training = int(sys.argv[2])
nr_forecasting = int(sys.argv[3])

def GetFile(fnombre):
    location = './data/' + fnombre
    print "locat:",location
    df = pd.read_csv(location)
    df['OC'] =  df['Open'] - df['Close']
    df['Title'] = fnombre
    #df['Date'] = pd.to_datetime(df['Date'])
    #df = df.sort_values(['Date'])
    df = df.set_index('Date')
    #afterd = pd.Timestamp(start_date)+pd.DateOffset(days_training)+pd.DateOffset(nr_forecasting)-pd.DateOffset(1)
    df = df.truncate(before=str(pd.Timestamp(start_date)-pd.DateOffset(1)))# after=str(afterd))
    df = df.iloc[0:days_training+nr_forecasting]
    df.to_csv('./single_data/' + fnombre)    
    return df

def PlotFile(fnombre):
    location = './single_data/' + fnombre
    df = pd.read_csv(location)
    df['Date'] = pd.to_datetime(df['Date'])

    if not df.empty:
        df = df.reset_index()
        df.plot(x='Date', y='OC', kind='line', title=fnombre, grid=1)
        fig = plt.gcf();
        fig.savefig('./single_data/' + fnombre +'.png', dpi = 100)
        plt.close() # don't shows plots
    
    return df;

def ArimaSingleFile(fnombre):
    location = './single_data/' + fnombre
    #print "location",location
    seriesOrig = pd.read_csv(location)
      
    # parametri ARIMA 
    
    #impostazione e calcolo del modello ARIMA
    series = seriesOrig[['Date', 'OC']]
    series = series.set_index('Date')
    X = series
    size = int(len(X))
    print "SIZE:",size

    train = X[0:(size-nr_forecasting)]
    #print train
    train = train.values

    print "TRAINING:",len(train)," DAYS"
    test = X[(size-nr_forecasting):size]
    #print test
    test = test.values
    print "TESTING:",len(test)
    #print "TESTING:",nr_forecasting," DAYS"

    history = [x for x in train]
    if len(history)==0:
        #print type(series)
        series = pd.DataFrame()
        series.to_csv('test/' + fnombre)
        return
    predictions = list()
    cont = 0
    p = 4
    d = 2
    q = 1

    errore = True
    while True:
        try:
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast(steps=len(test))#nr_forecasting)
            predictions = output[0].reshape(len(test),1)#nr_forecasting,1)
            errore = False
            break
        except:
            if errore==True:
                p = 0
                d = 0
                q = 0
                errore = False
            p = p + 1
            if p>5:
                p = 0
                d = d + 1
            if d>5:
                d=0
                q = q + 1
            if q>5:
                q = 0
            print "COMBINAZIONE ARIMA:",p,d,q
            print "ERRORE ARIMA fnom:",fnombre
            print sys.exc_info()
            #series = pd.DataFrame()
            #series.to_csv('test/' + fnombre)
            #return
    
    #print "PREDICTIONS:",predictions,type(predictions),predictions.shape
    series['Predicted'] = np.nan
    series.iloc[-len(test):, 1:2] = predictions
    series['Title'] = fnombre
    series['Expected_OC_perc'] = 0
    series['Predicted_OC_perc'] = 0

    seriesOrig = seriesOrig[['Date','Open','High','Low','Close','Adj Close','Volume','OC','Title']]
    seriesOrig = seriesOrig.set_index('Date')

    previous = None
    for date in series.T.iteritems():
        if previous!=None:
            den = seriesOrig.loc[date[0],'Open']
            series.loc[date[0],'Expected_OC_perc']=seriesOrig.loc[date[0],'OC']/den
            series.loc[date[0],'Predicted_OC_perc']=series.loc[date[0],'Predicted']/den
        previous = date
    #print "OK"

    series = series[(size-len(test)):size]
    series.to_csv('test/' + fnombre)
    
    # plot
    
    plt.plot(test)
    plt.plot(predictions, color='red')
    
    # plot data nelle ascisse (error: don't conver string to float)
    #plt.plot(series.index, test)
    #plt.plot(series.index, predictions, color='red')
    
    plt.title(fnombre)
    plt.legend(['expected', 'predicted'])
    plt.savefig('test/' + fnombre + '.png')
    #plt.show()
    plt.close()

def GetArimaSingleFile(fnombre):
    location = 'test/' + fnombre
    df = pd.read_csv(location)
    return df


if __name__ == "__main__":


    FileNames = []
    for files in os.listdir("./data/"):
        if files.endswith(".csv"):
            FileNames.append(files)

    #print len(FileNames)

    listFiles = [GetFile(file) for file in FileNames[:20]]

    df = [PlotFile(file) for file in FileNames[:20]]
    #print df[0].iloc[20]['Date']
    nextdate = df[0].iloc[20]['Date']
    #print type(df[0]),df[0]

    [ArimaSingleFile(file) for file in FileNames[:20]]

    #print "done"

    df = [GetArimaSingleFile(file) for file in FileNames[:20]]
    dftot = pd.concat(df)
    if dftot.empty==True:
        sys.exit(1)

    #print "dftot:",dftot
    #print "done"

    dftot = dftot.set_index(['Date']);


    #print "dftot:",dftot

    dftot['Expected_OC'] =  dftot['OC'];

    # Aggiunta della colonna Predicted_OC
    dftot['Predicted_OC'] =  dftot['Predicted'];
        
    dftot = dftot[['Title', 'Expected_OC', 'Predicted_OC', 'Expected_OC_perc','Predicted_OC_perc']]

    #print "dftot:",dftot
    dftot.to_csv('output/totale.csv')


    dftot_1 = dftot.sort_values(['Expected_OC_perc'], ascending=False); 
    #print "QUIDFTOT_1:",dftot_1
    migliori_exp = dftot_1.groupby('Date').nth((0,1,2,3,4)); 
    #print "MIGL:",migliori_exp
    migliori_exp.to_csv('output/expected_migliori.csv')

    # Calcolo dei 5 titoli peggiori sui risultati attesi (expected)
    dftot_2 = dftot.sort_values(['Expected_OC_perc'], ascending=True); 
    peggiori_exp = dftot_2.groupby('Date').nth((0,1,2,3,4)); 
    peggiori_exp.to_csv('output/expected_peggiori.csv')


    dftot_3 = dftot.sort_values(['Predicted_OC_perc'], ascending=False); 
    migliori_pred = dftot_3.groupby('Date').nth((0,1,2,3,4)); 
    migliori_pred.to_csv('output/predicted_migliori.csv')

    # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    dftot_4 = dftot.sort_values(['Predicted_OC_perc'], ascending=True); 
    peggiori_pred = dftot_4.groupby('Date').nth((0,1,2,3,4)); 
    peggiori_pred.to_csv('output/predicted_peggiori.csv')

    #print dftot
    dftot.reset_index(inplace=True)
    dftot['Date'] = pd.to_datetime(dftot['Date'])
    #print dftot

    incr_bydate_exp = migliori_exp.groupby('Date').sum()
    #print "incr_bydate_exp",incr_bydate_exp
    decr_bydate_exp = peggiori_exp.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    #print "decr_bydate_exp",decr_bydate_exp

    valore_giornaliero_exp = (incr_bydate_exp['Expected_OC_perc'] - decr_bydate_exp['Expected_OC_perc'])/10*100
    valore_giornaliero_exp.index = pd.to_datetime(valore_giornaliero_exp.index)

    # calcolo del valore percentuale sui valori predetti
    incr_bydate_pred = migliori_pred.groupby('Date').sum()
    decr_bydate_pred = peggiori_pred.groupby('Date').sum()
    print "MIGLIORIPRED:",migliori_pred
    print "CIAOINCRPRED:",incr_bydate_pred
    print "CIAODECRPRED:",decr_bydate_pred

    # peggiori da rendere negatvi (?)
    #decr_bydate_pred = (decr_bydate_pred)
    #print "incr_bydate_pred",incr_bydate_pred
    #print "decr_bydate_pred",decr_bydate_pred
    valore_giornaliero_pred = (incr_bydate_pred['Expected_OC_perc'] - decr_bydate_pred['Expected_OC_perc'])/10*100
    valore_giornaliero_pred.index = pd.to_datetime(valore_giornaliero_pred.index)

    print valore_giornaliero_exp.values.tolist()
    print valore_giornaliero_pred.values.tolist()
    print valore_giornaliero_exp.index.tolist()
    #print valore_giornaliero_pred.index.tolist()
    x_labstr = [el.strftime('%d-%m') for el in valore_giornaliero_exp.index.tolist()[0:nr_forecasting]]
    x_lab = np.arange(nr_forecasting)+1
    plt.plot(x_lab, valore_giornaliero_exp.values.tolist()[0:nr_forecasting])
    plt.plot(x_lab, valore_giornaliero_pred.values.tolist()[0:nr_forecasting])

    #xsexp = np.array(valore_giornaliero_exp.values.tolist()[0:nr_forecasting]).cumsum()
    #print "EXP:",xsexp
    #iexp = np.argmax(np.maximum.accumulate(xsexp) - xsexp)
    #if iexp==0:
    #    jexp = 0
    #else:
    #    jexp = np.argmax(xsexp[:iexp])

    xspred = np.array(valore_giornaliero_pred.values.tolist()[0:nr_forecasting]).cumsum() 
    ipred = np.argmax(np.maximum.accumulate(xspred) - xspred)
    if ipred==0:
        jpred = 0
    else:
        jpred = np.argmax(xspred[:ipred])

    #mddexp = xsexp[jexp]-xsexp[iexp]
    mddpred = xspred[jpred]-xspred[ipred]
    f = open('output/valori.txt','w+')
    f.write("return exp:"+str(sum(valore_giornaliero_exp.values.tolist()[0:nr_forecasting]))+"\n")

    f.write("return pred:"+str(sum(valore_giornaliero_pred.values.tolist()[0:nr_forecasting]))+"\n")
    f.write("mdd:"+str(mddpred)+"\n")
    f.write("return over maximum drawdown:"+str(sum(valore_giornaliero_pred.values.tolist()[0:nr_forecasting])/mddpred)+"\n")

    f.close()
    
    #print "QUIEXP:",x_lab,valore_giornaliero_exp.values.tolist()
    #print "QUIPRED:",x_lab,valore_giornaliero_pred.values.tolist()
    print "next command: python process.py",pd.Timestamp(nextdate).to_datetime().strftime('%Y-%m-%d'),"40 20"
    plt.title('Guadagni giornalieri')
    plt.xticks(np.arange(nr_forecasting)+1,x_labstr)
    plt.legend(['expected', 'predicted'])
    plt.savefig('output/valore_percentuale.png')
    plt.show()

    plt.title('Cuva del return')
    plt.legend(['Return'])
    plt.plot(xspred)
    plt.plot([ipred,jpred],[xspred[ipred],xspred[jpred]],'o',color='Red', markersize=10)
    plt.savefig('output/curvareturn.png')
    plt.show()
