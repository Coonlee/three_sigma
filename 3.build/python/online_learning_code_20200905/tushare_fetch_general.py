# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 02:53:24 2020

@author: suziqiao
"""

import numpy as np
import tushare as ts
import pandas as pd
from datetime import datetime
from random import choice

ts.set_token('e13aeb7d19e5189332ae3a602961d81e3dfbb56d6d13b4ed406d9f81')
pro = ts.pro_api()


def generate_sample_stock_lst(sample_size = 30, sample_frac = None, list_date_range = [None, None], delist_date_range = [None, None]):
    list_stock_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date,delist_date')
    delist_stock_df = pro.stock_basic(exchange='', list_status='D', fields='ts_code,list_date,delist_date')    
    all_stock_df = pd.concat([list_stock_df,delist_stock_df])
    
    if list_date_range[0] != None:
        all_stock_df = all_stock_df[all_stock_df.list_date >= list_date_range[0]]
    if list_date_range[1] != None:
        all_stock_df = all_stock_df[all_stock_df.list_date <= list_date_range[1]]
    if delist_date_range[0] != None:
        all_stock_df = all_stock_df[(all_stock_df.delist_date.isna() == True) | (all_stock_df.delist_date >= delist_date_range[0])]
    if delist_date_range[1] != None:
        all_stock_df = all_stock_df[(all_stock_df.delist_date.isna() == False) & (all_stock_df.delist_date <= delist_date_range[1])] 
        
    sample_stock_df = all_stock_df.sample(n = sample_size, frac = sample_frac, replace=False, weights=None, random_state=None, axis=0)
    sample_stock_lst = sample_stock_df['ts_code'].unique().tolist()
    
    return sample_stock_lst


def get_market_df(stock_lst, adj='hfq',freq='D'):
    market_df = pd.DataFrame(columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close',
       'change', 'pct_chg', 'vol', 'amount'])
    for item in stock_lst:
        tmp_df = ts.pro_bar(ts_code=item, adj=adj, freq=freq)
        market_df = market_df.append(tmp_df,ignore_index=True)
    market_df['pct_chg'] = market_df['pct_chg']/100.0
    market_df.dropna(inplace = True)
        
    return market_df
      

def get_matrix_df(market_df, start_date = None, end_date = None, value_column = 'pct_chg', drop_na = False):
    matrix_df = market_df.pivot(index = 'trade_date', values = value_column, columns = 'ts_code').reset_index()
    
    if start_date != None:
        matrix_df = matrix_df[matrix_df.trade_date >= start_date]
    if end_date != None:
        matrix_df = matrix_df[matrix_df.trade_date <= end_date]
    if drop_na:
        matrix_df.dropna(how = 'any', axis = 1, inplace = True)
    
    matrix_df.sort_values(by = ['trade_date'], inplace = True)
    
    return matrix_df


def generate_random_backtest_period(length_of_years = 5, earliest_start_year = 2005):
    year_lst = np.arange(earliest_start_year,datetime.now().year - length_of_years + 1, 1)
    month_lst = np.arange(101,1300,100)
    start_date = choice(year_lst)*10000+ choice(month_lst)
    end_date = start_date+10000*length_of_years
    return str(start_date), str(end_date)




#待优化事项：
    #1.处理开盘涨跌停
    #2.处理停复牌（通过同花顺确认tushare停复牌的字段含义与用法）
    #df = pro.suspend(ts_code='600030.SH', suspend_date='', resume_date='', fields='')