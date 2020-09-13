# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:07:51 2020

@author: suziqiao
"""

import numpy as np
import pandas as pd
from datetime import datetime 
import random
import copy
import matplotlib.pyplot as plt
import os
import math
from tushare_fetch_general import get_matrix_df, generate_random_backtest_period
from random import choice

#每个PositionInfo的对象，记录持仓的每一只股票的相关信息与状态，此对象的集合会以list的方式存在ManagementAccount对象里
class PositionInfo:
    def __init__(self, positionID, tickerSymbol, marketPrice, totalValue, amount, status, dailyReturn):
        self.positionID = positionID
        self.tickerSymbol = tickerSymbol
        self.marketPrice = marketPrice
        self.totalValue = totalValue
        self.amount = amount
        self.status = status
        self.dailyReturn = dailyReturn
        
    def updateMarketPrice(self, newMarketPrice):
        self.marketPrice = newMarketPrice       
    def updateAmount(self, updateAmount):
        self.amount = self.amount + updateAmount
    def updateStatus(self, newStatus):
        self.status = newStatus 
    def updateDailyReturn(self, newDailyReturn):
        self.dailyReturn = newDailyReturn
    def updateTotalValue(self):
        self.totalValue = self.marketPrice*self.amount
        

#此类模拟交易的账户，回测过程中，每个交易日生成一个账户的copy，来记录账户在每个交易日的时点信息
class ManagementAccount:
    def __init__(self, tradeDate, dailyReturn, netValue, totalAsset, totalCash, positionInfoList = []):
        self.tradeDate = tradeDate
        self.dailyReturn = dailyReturn
        self.netValue = netValue 
        self.totalAsset = totalAsset
        self.totalCash = totalCash
        self.positionInfoList = positionInfoList        
    def updateTradeDate(self, newTradeDate):
        self.tradeDate = newTradeDate
    def updateDailyReturn(self, newDailyReturn):
        self.dailyReturn = newDailyReturn
    def updateNetValue(self, newNetValue):
        self.netValue = newNetValue
    def updateTotalCash(self, deltaCash):
        self.totalCash += deltaCash
    def updateTotalAsset(self):
        newTotalAsset = self.totalCash
        for item in self.positionInfoList:
            newTotalAsset += item.totalValue
        self.totalAsset = newTotalAsset
    def updatePositionInfoList(self, newPositionInfoList):
        self.positionInfoList = newPositionInfoList
        
    def buyTrade(self, tickerSymbol, buyCashAmount, marketPrice, minTradeUnit, TransCostRate):
        if buyCashAmount > self.totalCash:
            buyCashAmount = self.totalCash
        inList = False
        for item in self.positionInfoList:
            if item.tickerSymbol == tickerSymbol:
                inList = True
                item.updateMarketPrice(marketPrice)
                buyShareAmount = buyCashAmount/(item.marketPrice*(1 + TransCostRate))
                buyShareAmount = math.floor(buyShareAmount/minTradeUnit)*minTradeUnit
                cashPayed = buyShareAmount*item.marketPrice*(1 + TransCostRate)
                item.updateAmount(buyShareAmount)
                item.updateTotalValue()
                self.updateTotalCash(-cashPayed)
        if inList == False:
            newPosition = PositionInfo(-1, tickerSymbol, marketPrice, 0.0, 0.0, 1, None)
            buyShareAmount = buyCashAmount/(newPosition.marketPrice*(1 + TransCostRate))
            buyShareAmount = math.floor(buyShareAmount/minTradeUnit)*minTradeUnit
            cashPayed = buyShareAmount*newPosition.marketPrice*(1 + TransCostRate)
            newPosition.updateAmount(buyShareAmount)
            newPosition.updateTotalValue()
            self.positionInfoList.append(newPosition)
            self.updateTotalCash(-cashPayed)
            
    def sellTrade(self, tickerSymbol, sellShareAmount, marketPrice, TransCostRate):
        inList = False
        for item in self.positionInfoList:
            if item.tickerSymbol == tickerSymbol:
                inList = True
                if item.amount <= sellShareAmount:
                    sellShareAmount = item.amount 
                item.updateAmount(-sellShareAmount)
                item.updateMarketPrice(marketPrice)
                cashReceived = sellShareAmount*marketPrice*(1 - TransCostRate)
                self.updateTotalCash(cashReceived)
                item.updateTotalValue()
        if inList == False:
            print("No position to sell!")

            
def dataPreprocess(backtest_market_df, backtest_length_of_years, expert_num_choice_pool):
    #Generate random start and end time for each of the backtest iterations
    start_date, end_date = generate_random_backtest_period(length_of_years = backtest_length_of_years)
    
    #Get the pivot dataframe regarding daily return, open price and close price
    return_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                              end_date = end_date, value_column = 'pct_chg', drop_na = False)
    open_price_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                  end_date = end_date, value_column = 'open', drop_na = False)
    close_price_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                   end_date = end_date, value_column = 'close', drop_na = False)
    uplimit_flag_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                   end_date = end_date, value_column = 'uplimit_flag', drop_na = False)
    downlimit_flag_df = get_matrix_df(backtest_market_df, start_date = start_date, 
                                   end_date = end_date, value_column = 'downlimit_flag', drop_na = False)
    
    for item in return_df.columns[1:]:
        if np.isnan(return_df[item].values[0]):
            return_df.drop(columns = [item], inplace = True)
    for item in open_price_df.columns[1:]:
        if np.isnan(open_price_df[item].values[0]):
            open_price_df.drop(columns = [item], inplace = True)
    for item in close_price_df.columns[1:]:
        if np.isnan(close_price_df[item].values[0]):
            close_price_df.drop(columns = [item], inplace = True)
    for item in uplimit_flag_df.columns[1:]:
        if np.isnan(uplimit_flag_df[item].values[0]):
            uplimit_flag_df.drop(columns = [item], inplace = True)
    for item in downlimit_flag_df.columns[1:]:
        if np.isnan(downlimit_flag_df[item].values[0]):
            downlimit_flag_df.drop(columns = [item], inplace = True)
    
    return_df.fillna(0.0, inplace = True)
    open_price_df.fillna(method = 'ffill', inplace = True)
    close_price_df.fillna(method = 'ffill', inplace = True)
    uplimit_flag_df.fillna(0.0, inplace = True)
    downlimit_flag_df.fillna(0.0, inplace = True)
    
    return_df.reset_index(inplace = True)
    return_df.drop(columns = ['index'],inplace = True)
    open_price_df.reset_index(inplace = True)
    open_price_df.drop(columns = ['index'],inplace = True)
    close_price_df.reset_index(inplace = True)
    close_price_df.drop(columns = ['index'],inplace = True)
    uplimit_flag_df.reset_index(inplace = True)
    uplimit_flag_df.drop(columns = ['index'],inplace = True)
    downlimit_flag_df.reset_index(inplace = True)
    downlimit_flag_df.drop(columns = ['index'],inplace = True)
    
    return_df['trade_date'] = return_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    open_price_df['trade_date'] = open_price_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    close_price_df['trade_date'] = close_price_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    uplimit_flag_df['trade_date'] = uplimit_flag_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    downlimit_flag_df['trade_date'] = downlimit_flag_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    
    expert_num_limit = choice(expert_num_choice_pool)
    if len(return_df.columns) > expert_num_limit:
        sample_expert_lst = random.sample(list(return_df.columns[1:]),expert_num_limit)
        sample_columns = [return_df.columns[0]]
        sample_columns.extend(sample_expert_lst)
        return_df = return_df[sample_columns]
        open_price_df = open_price_df[sample_columns]
        close_price_df = close_price_df[sample_columns]
        uplimit_flag_df = uplimit_flag_df[sample_columns]
        downlimit_flag_df = downlimit_flag_df[sample_columns]
    
    return return_df, open_price_df, close_price_df, uplimit_flag_df, downlimit_flag_df


#backtest process
def backtestProcess(backtestInputDf, symbolDailyReturnDf, symbolOpenPriceDf, symbolClosePriceDf, uplimit_flag_df, downlimit_flag_df, initial_cash = 1000000.0, minTradeUnit = 100, rebalanceFreq = 1,rebalanceThreshold = 0.05, TransCostRate = 0.003): 
    #backtestInputDf来自在线学习的输出，backtestSymbolDailyReturnList是每日每个股票的收益率的dataframe
    managementAccountList = []
    backtestStartDate = backtestInputDf['trade_date'].values[0]
    #Create a parallel account for the strategy of BAH(buy-and-hold), which demonstrates the average scenario.
    managementAccountList_BAH = []
    
    #Initialization for the first day
    managementAccount = ManagementAccount(backtestStartDate, 0, 1.0, initial_cash, initial_cash, [])
    
    for item in backtestInputDf.columns[1:]:
        positionInfo = PositionInfo(-1, item, None, 0.0, 0, 1, None)
        managementAccount.positionInfoList.append(positionInfo)
        purchasePositionCash = initial_cash/len(backtestInputDf.columns[1:])
        #第一天开盘时未涨停，则执行买入操作：
        if uplimit_flag_df[item].values[0] == 0:
            managementAccount.buyTrade(item, purchasePositionCash, symbolClosePriceDf[item].values[0], minTradeUnit, TransCostRate)
    managementAccount.updateTotalAsset()
    managementAccountCopy = copy.deepcopy(managementAccount)
    managementAccountList.append(managementAccountCopy)
    #Initialization of the buy-and-hold account  
    managementAccount_BAH = ManagementAccount(backtestStartDate, 0, 1.0, initial_cash, initial_cash, [])
    for item in backtestInputDf.columns[1:]:
        positionInfo = PositionInfo(-1, item, None, 0.0, 0, 1, None)
        managementAccount_BAH.positionInfoList.append(positionInfo)
        purchasePositionCash = initial_cash/len(backtestInputDf.columns[1:])
        #第一天开盘时未涨停，则执行买入操作：
        if uplimit_flag_df[item].values[0] == 0:
            managementAccount_BAH.buyTrade(item, purchasePositionCash, symbolClosePriceDf[item].values[0], minTradeUnit, TransCostRate)
    managementAccount_BAH.updateTotalAsset()
    managementAccountCopy_BAH = copy.deepcopy(managementAccount_BAH)
    managementAccountList_BAH.append(managementAccountCopy_BAH)
    
    #Iterate from the second day:
    for tradeDateIndex in range(1,len(backtestInputDf['trade_date'])):
        #Update managementAccount at the opening
        for item in managementAccount.positionInfoList:
            item.updateMarketPrice(symbolOpenPriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue()
        managementAccount.updateTotalAsset()
        #Calculate how much to sell and buy
        targetWeightVector = np.array(backtestInputDf[backtestInputDf.trade_date == backtestInputDf['trade_date'].values[tradeDateIndex - 1]][backtestInputDf.columns[1:]])[0]
        currentWeightVector = []
        for item in managementAccount.positionInfoList:
            currentWeightVector.append(item.totalValue)
        currentWeightVector = currentWeightVector/managementAccount.totalAsset
        deltaWeightVector = targetWeightVector - currentWeightVector 
        #Rebalancing conditions:
        if tradeDateIndex % rebalanceFreq == 0 or max(deltaWeightVector) > rebalanceThreshold:
            #Sell first in order to get sufficient cash to buy 
            for index in range(len(deltaWeightVector)):
                if deltaWeightVector[index] < 0:
                    deltaAmount = deltaWeightVector[index]*managementAccount.totalAsset/managementAccount.positionInfoList[index].marketPrice
                    #达到最小交易单元（一手），且当天开盘时未跌停，且当天未停牌，则执行卖出操作：
                    if math.ceil(deltaAmount/minTradeUnit) <= -1 and downlimit_flag_df[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] == 0 and (symbolOpenPriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] != symbolOpenPriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex - 1] or 
                         symbolClosePriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] != symbolClosePriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex - 1]):
                        managementAccount.sellTrade(managementAccount.positionInfoList[index].tickerSymbol, - math.ceil(deltaAmount/minTradeUnit)*100, managementAccount.positionInfoList[index].marketPrice, TransCostRate)
            #Buy after selling is done
            for index in range(len(deltaWeightVector)):
                if deltaWeightVector[index] > 0:
                    deltaAmount = deltaWeightVector[index]*managementAccount.totalAsset/managementAccount.positionInfoList[index].marketPrice
                    #达到最小交易单元（一手），且当天开盘时未涨停，且当天未停牌，则执行买入操作：
                    if math.floor(deltaAmount/minTradeUnit) >= 1 and uplimit_flag_df[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] == 0 and (symbolOpenPriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] != symbolOpenPriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex - 1] or 
                         symbolClosePriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex] != symbolClosePriceDf[managementAccount.positionInfoList[index].tickerSymbol].values[tradeDateIndex - 1]):
                        managementAccount.buyTrade(managementAccount.positionInfoList[index].tickerSymbol, deltaWeightVector[index]*managementAccount.totalAsset, managementAccount.positionInfoList[index].marketPrice, minTradeUnit, TransCostRate)
                        
        #Update positionInfoList again after the market closes
        for item in managementAccount.positionInfoList:
            item.updateMarketPrice(symbolClosePriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue() 
        #Update positionInfoList of the buy-and-hold account as well
        for item in managementAccount_BAH.positionInfoList:
            item.updateMarketPrice(symbolClosePriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue() 
        
        #Update managementAccount
        managementAccount.updateTradeDate(backtestInputDf['trade_date'].values[tradeDateIndex])
        managementAccount.updateTotalAsset()
        managementAccount.updateDailyReturn((managementAccount.totalAsset - managementAccountList[-1].totalAsset)/managementAccountList[-1].totalAsset)
        managementAccount.updateNetValue(managementAccountList[-1].netValue*(1.0 + managementAccount.dailyReturn))
        managementAccountCopy = copy.deepcopy(managementAccount)
        managementAccountList.append(managementAccountCopy)
        #Update the buy-and-hold managementAccount: managementAccount_BAH
        managementAccount_BAH.updateTradeDate(backtestInputDf['trade_date'].values[tradeDateIndex])
        managementAccount_BAH.updateTotalAsset()
        managementAccount_BAH.updateDailyReturn((managementAccount_BAH.totalAsset - managementAccountList_BAH[-1].totalAsset)/managementAccountList_BAH[-1].totalAsset)
        managementAccount_BAH.updateNetValue(managementAccountList_BAH[-1].netValue*(1.0 + managementAccount_BAH.dailyReturn))
        managementAccountCopy_BAH = copy.deepcopy(managementAccount_BAH)
        managementAccountList_BAH.append(managementAccountCopy_BAH)
        
    return managementAccountList, managementAccountList_BAH


def plotBacktestResult(return_df, close_netvalue_df, backtestResult, annualized_excess_return, path, figname = None):
    latest_netvalue_lst = close_netvalue_df.iloc[-1,1:].tolist()
    best_exp_index = latest_netvalue_lst.index(max(latest_netvalue_lst))
    best_exp_df = close_netvalue_df.iloc[:,[0,best_exp_index + 1]]
    best_exp_df.columns = ['trade_date','net_value']
    benchmark_df = copy.deepcopy(best_exp_df)
    benchmark_df['net_value'] = 1
    strategy_df = []
    best_exp_weight_df = []
    for item in backtestResult[0]:
        tmp_lst = []
        tmp_lst.append(item.tradeDate)
        tmp_lst.append(item.netValue)
        strategy_df.append(tmp_lst)
        best_exp_weight_df.append([item.tradeDate,item.positionInfoList[best_exp_index].totalValue/item.totalAsset])
        #best_exp_weight_df.append([item.tradeDate,item.positionInfoList[best_exp_index].amount])
    strategy_df = pd.DataFrame(strategy_df, columns = ['trade_date','net_value'])
    best_exp_weight_df = pd.DataFrame(best_exp_weight_df, columns = ['trade_date','weight'])  
    
    avg_exp_df = []
    for item in backtestResult[1]:
        tmp_lst = []
        tmp_lst.append(item.tradeDate)
        tmp_lst.append(item.netValue)
        avg_exp_df.append(tmp_lst)
    avg_exp_df = pd.DataFrame(avg_exp_df, columns = ['trade_date','net_value'])
    
    #Plot
    plt.figure(figsize = (30,15))
    plt.plot(strategy_df['trade_date'],strategy_df['net_value'], linewidth = 8, alpha = 1, color = 'blue')
    plt.plot(avg_exp_df['trade_date'],avg_exp_df['net_value'], linewidth = 6, alpha = 0.8, color = 'orangered')
    plt.plot(best_exp_df['trade_date'],best_exp_df['net_value'], linewidth = 4, alpha = 0.7, color = 'green')
    
    #plt.plot(benchmark_df['trade_date'],benchmark_df['net_value'], linewidth = 3, alpha = 0.7)
    for item in close_netvalue_df.columns[1:]:
        plt.plot(close_netvalue_df['trade_date'],close_netvalue_df[item], 
                 linewidth = 1, linestyle = '--', alpha = 0.7)
        
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel('Date',fontsize = 25)
    plt.ylabel('Net Value', fontsize = 25)
    plt.legend(['Strategy Line','Average Line','Best Expert Line'], loc = 'upper left', fontsize = 25)
    plt.twinx()
    plt.yticks(fontsize = 20)
    plt.ylabel('Weight', fontsize = 25)
    plt.plot(best_exp_weight_df['trade_date'],best_exp_weight_df['weight'], 
                 linewidth = 3, linestyle = '--', alpha = 0.7, color = 'black')
    plt.legend(['Best Expert Weight Line'], loc = 'upper right', fontsize = 25)
    
    #plt.title('Online Learning Backtest Result(LearningRate: ' + str(LearningRate) + '  TransCostRate: ' + str(TransCostRate) +
    #          '  RegVal: ' + str(RegVal) + '  Beta: ' + str(Beta) + '  rebalanceFreq: ' + str(rebalanceFreq) + '  rebalanceThreshold: ' + str(rebalanceThreshold) + ')', fontsize = 30)
    plt.title('Online Learning Backtest Result (annualized_excess_return: ' + str(round(annualized_excess_return*100,2)) + '%)', fontsize = 30)
    
    plt.grid()
    #plt.savefig('C:/Users/szq13821/Desktop/Online Learning/Online Learning Backtest Result.jpg')
    if figname != None:
        plt.savefig(path + '\\backtest result pic\\' + figname + '.jpg')
    plt.show()