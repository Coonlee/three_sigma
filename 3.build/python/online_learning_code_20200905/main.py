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
    
    LearningRate = 1
    TransCostRate = 0.003
    RegVal = 0.5
    Beta = 10
    rebalanceFreq = 1
    rebalanceThreshold = 0.05
    backtest_length_of_years = 5
    expert_num_limit = 20
    expert_num_choice_pool = [x for x in range(5, 21)]
    
    #注意：这个函数是在调用tushare接口拉行情数据，sample_size大时会耗时较长，
    #切记不要放入循环中频繁调取，因为这会过度使用tushare资源，我怕被Jimmy限制账号。。。
    backtest_stock_lst = generate_sample_stock_lst(sample_size = 300)
    #Get the market data of the backtest_stock_lst
    backtest_market_df = get_market_df(backtest_stock_lst)
    
    stats_lst = []
  
    for test_round in range(0,100):
        try:
            #Generate random start and end time for each of the backtest iterations
            start_date, end_date = generate_random_backtest_period(length_of_years = backtest_length_of_years)
            
            #Get the pivot dataframe regarding daily return, open price and close price
            return_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                      end_date = end_date, value_column = 'pct_chg', drop_na = False)
            open_price_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                          end_date = end_date, value_column = 'open', drop_na = False)
            close_price_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                           end_date = end_date, value_column = 'close', drop_na = False)
            
            for item in return_df.columns[1:]:
                if np.isnan(return_df[item].values[0]):
                    return_df.drop(columns = [item], inplace = True)
                    
            for item in open_price_df.columns[1:]:
                if np.isnan(open_price_df[item].values[0]):
                    open_price_df.drop(columns = [item], inplace = True)
            
            for item in close_price_df.columns[1:]:
                if np.isnan(close_price_df[item].values[0]):
                    close_price_df.drop(columns = [item], inplace = True)
            
            return_df.fillna(0.0, inplace = True)
            open_price_df.fillna(method = 'ffill', inplace = True)
            close_price_df.fillna(method = 'ffill', inplace = True)
            
            return_df.reset_index(inplace = True)
            return_df.drop(columns = ['index'],inplace = True)
            open_price_df.reset_index(inplace = True)
            open_price_df.drop(columns = ['index'],inplace = True)
            close_price_df.reset_index(inplace = True)
            close_price_df.drop(columns = ['index'],inplace = True)
             
            #Letting the value of the first day equal one
            for item in close_price_df.columns[1:]:
                close_price_df[item] = close_price_df[item].apply(lambda x: x/close_price_df[item].iloc[0])
            for item in open_price_df.columns[1:]:
                open_price_df[item] = open_price_df[item].apply(lambda x: x/open_price_df[item].iloc[0])
                
            return_df['trade_date'] = return_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
            open_price_df['trade_date'] = open_price_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
            close_price_df['trade_date'] = close_price_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
            
            expert_num_limit = choice(expert_num_choice_pool)
            if len(return_df.columns) > expert_num_limit:
                sample_expert_lst = random.sample(list(return_df.columns[1:]),expert_num_limit)
                sample_columns = [return_df.columns[0]]
                sample_columns.extend(sample_expert_lst)
                return_df = return_df[sample_columns]
                open_price_df = open_price_df[sample_columns]
                close_price_df = open_price_df[sample_columns]
            
            weight_storage_df = onlineLearningAlgo(return_df, LearningRate, RegVal, Beta, version = 'logarithmic')
            backtestResult = backtestProcess(weight_storage_df, return_df, open_price_df, close_price_df, rebalanceFreq, rebalanceThreshold, TransCostRate)  
            
            annualized_excess_return = pow(backtestResult[0][-1].netValue, 1.0/backtest_length_of_years) - pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years)
            average_annualized_return = pow(backtestResult[1][-1].netValue, 1.0/backtest_length_of_years) - 1.0
            stats_lst.append([expert_num_limit, average_annualized_return, annualized_excess_return])
            
            plotBacktestResult(return_df, close_price_df, backtestResult, annualized_excess_return, path, 'logarithmic_onlineLearning_result' + '_Round' + str(test_round + 1))
            
        except:
            continue
        
    stats_lst = pd.DataFrame(stats_lst, columns = ['expert_num_limit','average_annualized_return','annualized_excess_return'])
    stats_lst.to_excel('backtest_stats.xlsx', index = False)