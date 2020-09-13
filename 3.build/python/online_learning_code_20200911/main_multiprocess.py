# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:04:07 2020

@author: suziqiao
"""

#from tushare_fetch_general import get_matrix_df, generate_random_backtest_period
from online_algo import *
from backtest_framework import *
from general import execute_select,execute_dml,execute_ddl,write_table_to_sql 
from sqlalchemy import create_engine 
import sys 
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
from multiprocessing import Manager
import itertools
from random import choice


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


def backtest_process(para):
    RegVal = para[0]
    Beta = para[1]/para[0]
    stats_lst = []
    for test_round in range(0, 50):
        try:
            return_df, open_price_df, close_price_df, uplimit_flag_df, downlimit_flag_df = dataPreprocess(backtest_market_df, backtest_length_of_years, expert_num_choice_pool)
            weight_storage_df = onlineLearningAlgo(return_df, LearningRate, RegVal, Beta, version = 'logarithmic')
            backtestResult = backtestProcess(weight_storage_df, return_df, open_price_df, close_price_df, uplimit_flag_df, downlimit_flag_df, initial_cash, minTradeUnit, rebalanceFreq, rebalanceThreshold, TransCostRate)  
            
            annualized_excess_return = pow(backtestResult[0][-1].netValue, 1.0/backtest_length_of_years) - pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years)
            average_annualized_return = pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years) - 1.0
            stats_lst.append([expert_num_limit, average_annualized_return, annualized_excess_return])
            
            #Letting the value of the first day equal one
            #for item in close_price_df.columns[1:]:
                #close_price_df[item] = close_price_df[item].apply(lambda x: x/close_price_df[item].iloc[0])
            
            #plotBacktestResult(return_df, close_price_df, backtestResult, annualized_excess_return, path, 'logarithmic_onlineLearning_result' + '_Round' + str(test_round + 1))
        except:
            continue
    
    stats_lst = pd.DataFrame(stats_lst, columns = ['expert_num_limit','average_annualized_return','annualized_excess_return'])
    mean_excess = stats_lst['annualized_excess_return'].mean()
    std_excess = stats_lst['annualized_excess_return'].std()
    sharpe_excess = mean_excess/std_excess
    result_lst = [[para[0], para[1]/para[0], mean_excess, std_excess, sharpe_excess]]
    result_lst = pd.DataFrame(result_lst, columns = ['regval','beta','mean_excess','std_excess','sharpe_excess'])
    
    write_table_to_sql(result_lst, 'F_DOM_PRI_FUND_CANDIDATE_EXPERT2', 'ANA_FES_QUANT_STAGE', conn, onetime_load_num = 1)


connect_info = 'mysql+pymysql://da_admin:da_admin!@#123@10.2.160.60:3306/ANA_FES_QUANT_STAGE?charset=utf8'  
conn = create_engine(connect_info, pool_recycle = 300)  # use sqlalchemy to build link-engine
connection = conn.connect()

if __name__ == '__main__':
    #Get an overall sample stock pool for the entire backtest iterations
    #path = os.getcwd()
    minTradeUnit = 100
    initial_cash = 1000000.0
    LearningRate = 1
    TransCostRate = 0.002
    rebalanceFreq = 1
    rebalanceThreshold = 0.05
    backtest_length_of_years = 5
    expert_num_choice_pool = [x for x in range(8, 21)]
    
    #注意：这个函数是在调用tushare接口拉行情数据，sample_size大时会耗时较长，
    #切记不要放入循环中频繁调取，因为这会过度使用tushare资源，我怕被Jimmy限制账号。。。
    #backtest_stock_lst = generate_sample_stock_lst(sample_size = 300)
    #Get the market data of the backtest_stock_lst
    #backtest_market_df = get_market_df(backtest_stock_lst)
    #write_table_to_sql(backtest_market_df, 'F_DOM_PRI_FUND_CANDIDATE_SOURCE', 'ANA_FES_QUANT_STAGE', conn, onetime_load_num = 100000)
    
    backtest_market_sql = """select * from ANA_FES_QUANT_STAGE.F_DOM_PRI_FUND_CANDIDATE_SOURCE"""
    backtest_market_df = pd.read_sql(backtest_market_sql, conn)
    backtest_market_df['trade_date'] = backtest_market_df['trade_date'].apply(lambda x: str(x)) 
    
    backtest_market_df.sort_values(by = ['ts_code','trade_date'], inplace = True)
    #backtest_market_df['pre_close'] = backtest_market_df.groupby(['ts_code'])['pre_close'].transform(lambda x: x.bfill())
    backtest_market_df['uplimit_flag'] = backtest_market_df.apply(lambda x: 1 if x.open/x.pre_close > 1.095 else 0, axis = 1)
    backtest_market_df['downlimit_flag'] = backtest_market_df.apply(lambda x: 1 if x.open/x.pre_close < 0.905 else 0, axis = 1)
    
    #RegVal = np.arange(1,21,1)
    #Beta = np.arange(2,21,2)
    #combo_lst = list(itertools.product(RegVal,Beta))
    
    RegVal = [0.25,0.5,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50]
    para_product = [1,2,4,8,16,32,64,128,256,512]
    combo_lst = list(itertools.product(RegVal,para_product))
     
    #manager = Manager()
    #multiprocess_lst = manager.list()  
    pool = ProcessPool(8)
    pool.map(backtest_process, combo_lst)
    pool.close()
    pool.join()
    #multiprocess_lst = list(multiprocess_lst)
    #multiprocess_lst = pd.DataFrame(multiprocess_lst, columns = ['regval','beta','mean_excess','std_excess','sharpe_excess'])
    
    #write_table_to_sql(multiprocess_lst, 'F_DOM_PRI_FUND_CANDIDATE_EXPERT', 'ANA_FES_QUANT_STAGE', conn, onetime_load_num = 100000)
