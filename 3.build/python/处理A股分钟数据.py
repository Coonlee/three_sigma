# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:11:47 2020

@author: admin
"""



import pandas as pd
import numpy as np
import re
from datetime import datetime
import time
import pyarrow as pa
import pyarrow.parquet as pq




"""处理A股分钟数据"""
calc_start_time = time.time()

#load
#df = pd.read_csv('./data/201901_SH_sample.csv', encoding = "GBK", header=1).dropna()
#reader = pd.read_csv('../data/201901_SH_sample.csv', encoding = "GBK", header=1, chunksize=100)
reader = pd.read_csv('../data/201901_SH.csv', encoding = "GBK", header=1, chunksize=100000)


#clean
df = pd.DataFrame()
row_count = 0

for chunk in reader:    
    row_count = row_count + chunk.shape[0]
    chunk.dropna(inplace=True)

#    if row_count > 100000:  #for test
#        break
    
    for c in chunk.columns:
        try:
            chunk = chunk[c].str.split(',', expand=True)
        except:
            pass
    
    chunk.columns = ['Symbol',
                     'ShortName',
                     'TradingDate',
                     'TradingTime',
                     'TradingTime2',
                     'TradingTime3',
                     'TradingTime4',
                     'TradingTime5',
                     'OpenPrice',
                     'HighPrice',
                     'LowPrice',
                     'ClosePrice',
                     'Volumne',
                     'Amount',
                     'UNIX',
                     'Market',
                     'SecurityID',
                     'BenchmarkOpenPrice',
                     'Change',
                     'ChangeRatio',
                     'TotalVolume',
                     'VWAP',
                     'CumulativeLowPrice',
                     'CumulativeHighPrice',
                     'CumulativeVWAP',
                     'BuyPrice01',
                     'BuyPrice02',
                     'BuyPrice03',
                     'BuyPrice04',
                     'BuyPrice05',
                     'SellPrice01',
                     'SellPrice02',
                     'SellPrice03',
                     'SellPrice04',
                     'SellPrice05',
                     'BuyVolumn01',
                     'BuyVolumn02',
                     'BuyVolumn03',
                     'BuyVolumn04',
                     'BuyVolumn05',
                     'SellVolume01',
                     'SellVolume02',
                     'SellVolume03',
                     'SellVolume04',
                     'SellVolume05']
    for i in range(0, 45):
#        chunk.rename(columns={i:'col_'+str(i)}, inplace=True)
        chunk.iloc[:,i] = chunk.iloc[:,i].str.replace('(', '').str.replace(')', '').str.replace("'", '').str.replace('Decimal', '').str.replace('datetime.datetime', '').str.replace(' ', '')
#    chunk.iloc[:,i] = chunk.iloc[:,i].str.replace('(', '').str.replace(')', '').str.replace("'", '').str.replace('Decimal', '').str.replace('datetime.datetime', '').str.replace(' ', '')
    chunk['TradingTime'] = chunk['TradingTime'] + '-' + chunk['TradingTime2'] + '-' + chunk['TradingTime3'] + ' ' + chunk['TradingTime4'] + ':' + chunk['TradingTime5']
    chunk['TradingTime'] = chunk['TradingTime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
    chunk.drop(['TradingTime2','TradingTime3','TradingTime4','TradingTime5'], axis=1, inplace=True)
    chunk.reset_index(inplace=True)
    chunk.drop('index', axis=1, inplace=True)
    chunk = chunk[~chunk['ShortName'].apply(lambda x: re.match('沪深|中证|上证|股指|期货|ic|ih', x)).isnull()]
#    chunk = chunk[~chunk['ShortName'].apply(lambda x: re.match('四川|中远', x)).isnull()]
    df = df.append(chunk)
    print('\r Collected rows = ' + str(df.shape[0]) + ', 总耗时（秒数）：' + str(round(time.time() - calc_start_time)), end='')
    


#output
#df.to_excel('../data/201901_SH_adj_' + datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S') + '.xlsx', index=False)
df.to_excel('../data/201901_SH_adj.xlsx', index=False)
print('\n201901_SH_adj.xlsx was generated successfully.')






"""转parquet"""
#https://arrow.apache.org/docs/python/parquet.html
#import fastparquet

df = pd.read_excel('../data/201901_SH_adj.xlsx')

#fast parquet approach
#df2.to_parquet('./data/201901_SH_parquet.gzip', compression='gzip')


#pyarrow approach
table = pa.Table.from_pandas(df)
pq.write_table(table, '../data/201901_SH_adj.parquet')
#table2 = pq.read_table('../data/201901_SH.parquet')
#df2 = table2.to_pandas()
print('201901_SH_adj.parquet was generated successfully.')













