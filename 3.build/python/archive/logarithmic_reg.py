import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt.solvers import qp, options

NumOfExp = 5
NumOfDay = 100
TransCostRate = 0.003
RegVal = 0.5 ## Test several value
Beta = 10
# Assume return matrix is an nxT matrix. Each entry is log(return) and each column
# being list of log return on day t for n different stocks
return_matrix = np.random.normal(size=[NumOfExp, NumOfDay],loc=0.0, scale=0.03)

# Weight is stored as a column vector of size n where n is the number of experts
weight_storage = np.zeros((NumOfExp, NumOfDay))
aggressive_weight_storage = np.zeros((NumOfExp, NumOfDay))
weight = np.array([1 / NumOfExp] * NumOfExp)
A_t = np.eye(NumOfExp)*RegVal*2 ## Log online learning var
b_t = 0*np.ones(NumOfExp) ## Log online learning var
reg_mat = np.eye(NumOfExp)*RegVal


for i in range(NumOfDay):
    ## f(x) = -log(x^T r)
    last_return = np.sum(return_matrix[:, i]*weight)
    #问题：下面这个nabla_t是怎么推出来的？
    nabla_t = -1*last_return*return_matrix[:, i]
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
    sum_log_return = np.sum(return_matrix[:,:i], axis=1)
    fractional_weight = weight*np.exp(sum_log_return)
    normalize_frac_weight = weight
    aggressive_weight_storage[:,i] = normalize_frac_weight[:]

#pd.reset_option('display.float_format')
desc_weight = pd.DataFrame(weight)

# weight is the fraction of each expert on a time stamp
# We need to run backtesting according to weight


