# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:06:18 2020

@author: suziqiao
"""

import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp, options


def onlineLearningAlgo(return_df, LearningRate = 5, RegVal = 0.0001, Beta = 0.1, version = 'basic'):    
    return_matrix = return_df.iloc[:,1:]
    return_matrix = np.array(return_matrix).T                                                                                   
    NumOfDay = len(return_matrix[0])
    NumOfExp = len(return_matrix)  
    # Weight is stored as a column vector of size n where n is the number of experts
    weight_storage = np.zeros((NumOfExp, NumOfDay))
    aggressive_weight_storage = np.zeros((NumOfExp, NumOfDay))
    weight = np.array([1 / NumOfExp] * NumOfExp) 
    
    if version == 'basic':
        for i in range(NumOfDay):
            #Traditional
            delta_vector = 1+LearningRate*return_matrix[:, i]
            adjust_vector = weight * delta_vector
            weight =  adjust_vector/(np.sum(adjust_vector))
            weight_storage[:,i] = weight[:]
            #Aggressive
            sum_log_return = np.sum(return_matrix[:,:i], axis=1)
            fractional_weight = weight*np.exp(sum_log_return)
            normalize_frac_weight = fractional_weight/np.sum(fractional_weight)
            aggressive_weight_storage[:,i] = normalize_frac_weight[:]
            #pd.reset_option('display.float_format')  
            
    elif version == 'logarithmic':
        A_t = np.eye(NumOfExp)*RegVal*2 ## Log online learning var
        b_t = 0*np.ones(NumOfExp) ## Log online learning var
        reg_mat = np.eye(NumOfExp)*RegVal
        return_matrix = return_matrix + 1.0
        for i in range(NumOfDay):
            ## f(x) = -log(x^T r)
            last_return = np.sum(return_matrix[:, i]*weight)
            nabla_t = -1.0/last_return*return_matrix[:, i]
            A_t += np.outer(nabla_t,nabla_t)
            b_t += nabla_t*(np.inner(nabla_t,weight) - 1/Beta)
            y = np.dot(np.linalg.pinv(A_t), b_t)
        
            ## Solve x by a quadratic programming
            S = matrix(A_t + reg_mat)
            G = matrix(0.0, (NumOfExp, NumOfExp))
            G[::NumOfExp + 1] = -1.0
            h = matrix(0.0, (NumOfExp, 1))
            A = matrix(1.0, (1, NumOfExp))
            b = matrix(1.0)
            q = matrix(-2*y - 2*RegVal*weight)
            qp_weight = qp(S, q, G, h, A, b)['x'] #quadratic programming
        
            ## Store weight
            weight = np.array(qp_weight[:]).squeeze()
            weight_storage[:,i] = weight[:]
            #Aggressive
            #sum_log_return = np.sum(return_matrix[:,:i], axis=1)
            #fractional_weight = weight*np.exp(sum_log_return)
            #normalize_frac_weight = fractional_weight/np.sum(fractional_weight)
            normalize_frac_weight = weight
            aggressive_weight_storage[:,i] = normalize_frac_weight[:]
        
        #desc_weight = pd.DataFrame(weight)
        # weight is the fraction of each expert on a time stamp
        # We need to run backtesting according to weight
        #weight_storage_df = pd.concat([return_df[['trade_date']],pd.DataFrame(weight_storage.T, columns = return_df.columns[1:])], axis = 1)
    weight_storage_df = pd.concat([return_df[['trade_date']],pd.DataFrame(aggressive_weight_storage.T, columns = return_df.columns[1:])], axis = 1)      
        
    return weight_storage_df