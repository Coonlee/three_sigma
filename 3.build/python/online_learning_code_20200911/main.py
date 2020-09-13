# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:04:07 2020

@author: suziqiao
"""

from tushare_fetch_general import *
from online_algo import *
from backtest_framework import *


if __name__ == '__main__':
    #Get an overall sample stock pool for the entire backtest iterations
    path = os.getcwd()
    
    minTradeUnit = 100
    initial_cash = 1000000.0
    LearningRate = 1
    TransCostRate = 0.002
    RegVal = 10
    Beta = 10
    rebalanceFreq = 1
    rebalanceThreshold = 0.05
    backtest_length_of_years = 15
    expert_num_choice_pool = [x for x in range(8, 21)]
    
    #注意：这个函数是在调用tushare接口拉行情数据，sample_size大时会耗时较长，
    #切记不要放入循环中频繁调取，因为这会过度使用tushare资源，我怕被Jimmy限制账号。。。
    backtest_stock_lst = generate_sample_stock_lst(sample_size = 300)
    #Get the market data of the backtest_stock_lst
    backtest_market_df = get_market_df(backtest_stock_lst)
    
    backtest_market_df.sort_values(by = ['ts_code','trade_date'], inplace = True)
    #backtest_market_df['pre_close'] = backtest_market_df.groupby(['ts_code'])['pre_close'].transform(lambda x: x.bfill())
    backtest_market_df['uplimit_flag'] = backtest_market_df.apply(lambda x: 1 if x.open/x.pre_close > 1.095 else 0, axis = 1)
    backtest_market_df['downlimit_flag'] = backtest_market_df.apply(lambda x: 1 if x.open/x.pre_close < 0.905 else 0, axis = 1)

    stats_lst = []
    for test_round in range(0, 1):
        try:
            return_df, open_price_df, close_price_df, uplimit_flag_df, downlimit_flag_df = dataPreprocess(backtest_market_df, backtest_length_of_years, expert_num_choice_pool)
            
            weight_storage_df = onlineLearningAlgo(return_df, LearningRate, RegVal, Beta, version = 'logarithmic')
            backtestResult = backtestProcess(weight_storage_df, return_df, open_price_df, close_price_df, uplimit_flag_df, downlimit_flag_df, initial_cash, minTradeUnit, rebalanceFreq, rebalanceThreshold, TransCostRate)  
            
            annualized_excess_return = pow(backtestResult[0][-1].netValue, 1.0/backtest_length_of_years) - pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years)
            average_annualized_return = pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years) - 1.0
            stats_lst.append([return_df.shape[1] - 1, average_annualized_return, annualized_excess_return])
            
            #Letting the value of the first day equal one
            close_netvalue_df = copy.deepcopy(close_price_df)
            for item in close_netvalue_df.columns[1:]:
                close_netvalue_df[item] = close_netvalue_df[item].apply(lambda x: x/close_netvalue_df[item].iloc[0])
            
            plotBacktestResult(return_df, close_netvalue_df, backtestResult, annualized_excess_return, path, 'logarithmic_onlineLearning_result' + '_Round' + str(test_round + 1))  
        except:
            continue
        
    stats_lst = pd.DataFrame(stats_lst, columns = ['expert_num','average_annualized_return','annualized_excess_return'])
    mean_excess = round(stats_lst['annualized_excess_return'].mean(),3)
    std_excess = round(stats_lst['annualized_excess_return'].std(),3)
    plt.figure(figsize = (8,4))
    stats_lst['annualized_excess_return'].plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Distribution of Annualized Excess Return (Mean=' + str(mean_excess) + ',  Std=' + str(std_excess) + ')')
    plt.xlabel('Annualized Excess Return')
    plt.ylabel('Counts')
    plt.savefig('Distribution of Annualized Excess Return.jpg')
    plt.show()
    stats_lst.to_excel('backtest_stats.xlsx', index = False)