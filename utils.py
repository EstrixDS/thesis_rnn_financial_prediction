import yfinance as yf
import pandas as pd
import datetime

def acquisition(start: datetime.date,end: datetime.date,symbol_li: list):
    """generates the data for the predictions

    Args:
        start (datetime.date): date of the first stock data
        end (datetime.date): date of the last stock data
        symbol_li (list): list of stock ticker symbols listed on NASDAQ

    Returns: Dataframe with one row per date and stock
    """
    # create empty dataframe
    stock_final = pd.DataFrame()

    # iterate over each symbol
    for i in symbol_li:  
        
        # print the symbol which is being downloaded
        print(str(symbol_li.index(i)) + str(' : ') + i, sep=',',flush=True)  
        try:
            # download the stock price 
            stock = []
            stock = yf.download(i,start=start, end=end, progress=False)
                
            # append the individual stock prices 
            if len(stock) == 0:
                None
            else:
                stock['Name']=i
                stock_final = stock_final.append(stock,sort=False)
        except Exception:
            None
    return stock_final