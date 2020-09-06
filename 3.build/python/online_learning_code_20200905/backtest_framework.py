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
 


#此类每个对象模型的是出资人，在我们这个项目里不重要，可以忽略，使用时创建一个对象就行了
class FundProvider:
    def __init__(self, providerID, cashAmount, availableStartDate, availableEndDate, positionInfoList = []):
        self.providerID = providerID
        self.cashAmount = cashAmount
        self.availableStartDate = availableStartDate
        self.availableEndDate = availableEndDate
        self.positionInfoList = positionInfoList
    def updatePositionInfoList(self, newPositionInfoList):
        self.positionInfoList = newPositionInfoList

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
    def __init__(self, tradeDate, dailyReturn, netValue, totalAsset, positionInfoList = []):
        self.tradeDate = tradeDate
        self.dailyReturn = dailyReturn
        self.netValue = netValue 
        self.totalAsset = totalAsset
        self.positionInfoList = positionInfoList        
    def updateTradeDate(self, newTradeDate):
        self.tradeDate = newTradeDate
    def updateDailyReturn(self, newDailyReturn):
        self.dailyReturn = newDailyReturn
    def updateNetValue(self, newNetValue):
        self.netValue = newNetValue
    def updateTotalAsset(self, transactionCost):
        newTotalAsset = 0.0
        for item in self.positionInfoList:
            newTotalAsset += item.totalValue
        self.totalAsset = newTotalAsset - transactionCost
    def updatePositionInfoList(self, newPositionInfoList):
        self.positionInfoList = newPositionInfoList
        
      
#此对象描述交易这个动作                       
class Trade:
    def __init__(self, tradeID, tickerSymbol, tradeDate, tradeType, tradeAmount):
        self.tradeID = tradeID
        self.tickerSymbol = tickerSymbol
        self.tradeDate = tradeDate
        self.tradeType = tradeType
        self.tradeAmount = tradeAmount        
    def setTradeAmount(self, tradeAmount):
        self.tradeAmount = tradeAmount



def pivotSourceTable(file_name,start_date,end_date,return_type = 'ratio'):
    return_df = pd.read_excel(file_name)
    if return_df.columns[0] == 'Unnamed: 0':
        return_df.drop(columns = ['Unnamed: 0'], inplace = True)
    return_df['trade_date'] = return_df['trade_date'].apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
    return_df.drop_duplicates(subset = ['ts_code','trade_date'], inplace = True)
    return_df.sort_values(by = ['ts_code','trade_date'], inplace = True)
    #return_df = return_df[(return_df.ts_code != '300122.SZ') & (return_df.ts_code != '601888.SH') & (return_df.ts_code != '002714.SZ') & (return_df.ts_code != '600276.SH')]
    
    close_price_df = return_df.pivot(index = 'trade_date', values = 'close', columns = 'ts_code').reset_index()
    #close_price_df = close_price_df[close_price_df['trade_date'] > dt.date(2016,1,1)].dropna(axis = 1)
    close_price_df = close_price_df[(close_price_df['trade_date'] >= start_date)&(close_price_df['trade_date'] <= end_date)].dropna(axis = 1)
    open_price_df = return_df.pivot(index = 'trade_date', values = 'open', columns = 'ts_code').reset_index()
    #open_price_df = open_price_df[open_price_df['trade_date'] > dt.date(2016,1,1)].dropna(axis = 1)
    open_price_df = open_price_df[(open_price_df['trade_date'] >= start_date)&(open_price_df['trade_date'] <= end_date)].dropna(axis = 1)
    
    for item in close_price_df.columns[1:]:
        close_price_df[item] = close_price_df[item].apply(lambda x: x/close_price_df[item].iloc[0])
    for item in open_price_df.columns[1:]:
        open_price_df[item] = open_price_df[item].apply(lambda x: x/open_price_df[item].iloc[0])
    
    if return_type == 'ratio':
        return_df['ratio_return'] = return_df['pct_chg']/100.0
        return_df.fillna(0, inplace = True)
        return_df = return_df.pivot(index = 'trade_date', values = 'ratio_return', columns = 'ts_code').reset_index()
    else:
        return_df['log_return'] = return_df['pct_chg'].apply(lambda x: np.log(1.0 + x/100.0))
        return_df.fillna(0, inplace = True)
        return_df = return_df.pivot(index = 'trade_date', values = 'log_return', columns = 'ts_code').reset_index()
     
    #return_df = return_df_pivot[return_df_pivot['trade_date'] > dt.date(2016,1,1)].dropna(axis = 1)
    return_df = return_df[(return_df['trade_date'] >= start_date)&(return_df['trade_date'] <= end_date)].dropna(axis = 1)
    return_df = return_df.reset_index()
    return_df.drop(columns = ['index'], inplace = True)
    
    return return_df, open_price_df, close_price_df


#backtest process
def backtestProcess(backtestInputDf, symbolDailyReturnDf, symbolOpenPriceDf, symbolClosePriceDf, rebalanceFreq = 1,rebalanceThreshold = 0.05, TransCostRate = 0.003): 
    #backtestInputDf来自在线学习的输出，backtestSymbolDailyReturnList是每日每个股票的收益率的dataframe
    accountPositionInfoList = []
    managementAccountList = []
    initial_cash = 1000000.0
    cumuTransactionCost_lst = []
    backtestStartDate = backtestInputDf['trade_date'].values[0]
    
    #Create a parallel account for the strategy of buy-and-hold, which demonstrates the average scenario.
    accountPositionInfoList_BAH = []
    managementAccountList_BAH = []
     
    #Initialization for the first day
    managementAccount = ManagementAccount(backtestStartDate, 0, 1.0, initial_cash, accountPositionInfoList)
    for item in backtestInputDf.columns[1:]:
        positionInfo = PositionInfo(-1, item, symbolClosePriceDf[item].values[0], initial_cash/len(backtestInputDf.columns[1:]), initial_cash/len(backtestInputDf.columns[1:])/symbolOpenPriceDf[item].values[0], 1, symbolDailyReturnDf[item].values[0])
        accountPositionInfoList.append(positionInfo)
    managementAccount.updatePositionInfoList(accountPositionInfoList)
    managementAccountCopy = copy.deepcopy(managementAccount)
    managementAccountList.append(managementAccountCopy)
    
    #Initialize in the same way for buy-and-hold account:
    managementAccount_BAH = ManagementAccount(backtestStartDate, 0, 1.0, initial_cash, accountPositionInfoList_BAH)
    for item in backtestInputDf.columns[1:]:
        positionInfo_BAH = PositionInfo(-1, item, symbolClosePriceDf[item].values[0], initial_cash/len(backtestInputDf.columns[1:]), initial_cash/len(backtestInputDf.columns[1:])/symbolOpenPriceDf[item].values[0], 1, symbolDailyReturnDf[item].values[0])
        accountPositionInfoList_BAH.append(positionInfo_BAH)
    managementAccount_BAH.updatePositionInfoList(accountPositionInfoList_BAH)
    managementAccountCopy_BAH = copy.deepcopy(managementAccount_BAH)
    managementAccountList_BAH.append(managementAccountCopy_BAH)
    
    date_index = 0
    for tradeDateIndex in range(1,len(backtestInputDf['trade_date'])):
        transactionCost = 0.0
        #Update positionInfoList at the opening
        for item in managementAccount.positionInfoList:
            item.updateMarketPrice(symbolOpenPriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue()
        targetWeightVector = np.array(backtestInputDf[backtestInputDf.trade_date == backtestInputDf['trade_date'].values[tradeDateIndex - 1]][backtestInputDf.columns[1:]])[0]
        currentWeightVector = []
        for item in managementAccount.positionInfoList:
            currentWeightVector.append(item.totalValue)
        currentWeightVector = currentWeightVector/np.sum(currentWeightVector)
        deltaWeightVector = targetWeightVector - currentWeightVector 
        
        for index in range(len(deltaWeightVector)):
            if date_index % rebalanceFreq == 0 or max(deltaWeightVector) > rebalanceThreshold:
                managementAccount.positionInfoList[index].updateAmount(deltaWeightVector[index]*managementAccount.totalAsset/managementAccount.positionInfoList[index].marketPrice)
                transactionCost += abs(deltaWeightVector[index]*managementAccount.totalAsset)*TransCostRate        
        
        if len(cumuTransactionCost_lst) > 0:
            transactionCost += cumuTransactionCost_lst[-1]
        cumuTransactionCost_lst.append(transactionCost) 
        
        #Update positionInfoList again after the market closes
        for item in managementAccount.positionInfoList:
            item.updateMarketPrice(symbolClosePriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue() 
        #Update positionInfoList of the buy-and-hold account
        for item in managementAccount_BAH.positionInfoList:
            item.updateMarketPrice(symbolClosePriceDf[item.tickerSymbol].values[tradeDateIndex])
            item.updateTotalValue() 
        
        #Update managementAccount
        managementAccount.updateTradeDate(backtestInputDf['trade_date'].values[tradeDateIndex])
        managementAccount.updateTotalAsset(transactionCost)
        managementAccount.updateDailyReturn((managementAccount.totalAsset - managementAccountList[-1].totalAsset)/managementAccountList[-1].totalAsset)
        managementAccount.updateNetValue(managementAccountList[-1].netValue*(1.0 + managementAccount.dailyReturn))
        managementAccountCopy = copy.deepcopy(managementAccount)
        managementAccountList.append(managementAccountCopy)
        #Update the buy-and-hold managementAccount: managementAccount_BAH
        managementAccount_BAH.updateTradeDate(backtestInputDf['trade_date'].values[tradeDateIndex])
        managementAccount_BAH.updateTotalAsset(0)
        managementAccount_BAH.updateDailyReturn((managementAccount_BAH.totalAsset - managementAccountList_BAH[-1].totalAsset)/managementAccountList_BAH[-1].totalAsset)
        managementAccount_BAH.updateNetValue(managementAccountList_BAH[-1].netValue*(1.0 + managementAccount_BAH.dailyReturn))
        managementAccountCopy_BAH = copy.deepcopy(managementAccount_BAH)
        managementAccountList_BAH.append(managementAccountCopy_BAH)
        
        date_index += 1
        
    return managementAccountList, managementAccountList_BAH



def plotBacktestResult(return_df, close_price_df, backtestResult, annualized_excess_return, path, figname = None):
    latest_netvalue_lst = close_price_df.iloc[-1,1:].tolist()
    best_exp_index = latest_netvalue_lst.index(max(latest_netvalue_lst))
    best_exp_df = close_price_df.iloc[:,[0,best_exp_index + 1]]
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
    for item in close_price_df.columns[1:]:
        plt.plot(close_price_df['trade_date'],close_price_df[item], 
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

