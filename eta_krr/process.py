'''
Created on Jun 5, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import  numpy as np
from scipy import linalg

def solve_f(Q_arr, Lapl, y_vec_arr, reg_lambda):
    denom = linalg.inv(Q_arr.dot(Q_arr.T) + reg_lambda*Lapl)
    f_vec = denom.dot(Q_arr).dot(y_vec_arr)
    return f_vec

def slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    LOOCV_cost = 0
    for n in xrange(N): 
        _Q_leave_n = np.delete(Q_arr, n, 0)
        _y_leave_n = np.delete(y_vec_arr, n, 0)
        _f_vec_leave_n = solve_f(_Q_leave_n, Lapl, _y_leave_n, reg_lambda)
        LOOCV_cost += (y_vec_arr[n] - Q_arr[:, n].T.dot(_f_vec_leave_n))^2
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost 

def optimize_lambda(N, Q_arr, y_vec_arr, Lapl, 
                    min_lambda, max_lambda, increment, fast=True):
    LOOCV_argmin_lambda = np.inf
    for reg_lambda in xrange(min_lambda, max_lambda, increment):
        if fast:
            error = fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda)
        else:
            error = slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda)
        if error < LOOCV_argmin_lambda:
            LOOCV_argmin_lambda = reg_lambda
    return LOOCV_argmin_lambda

def fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    I = np.identity(N)
    inverted = linalg.inv(linalg.inv(Q_arr.dot(Q_arr.T) + reg_lambda*Lapl))
    H = Q_arr.T.dot(inverted.dot(Q_arr))
    I_minus_H = I - H
    mat= linalg.inv(linalg.block_diag(I_minus_H)).dot(I_minus_H.dot(y_vec_arr))
    LOOCV_cost = mat.T.dot(mat) 
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost

def predict_travel_time(optim_f_vec, Q_test_arr):
    return Q_test_arr.T.dot(optim_f_vec)

def validate(y_test_vec_arr, pred_y_vec_arr):
    diff = pred_y_vec_arr - y_test_vec_arr
    return diff

def build_model(N, Q_arr, y_vec, Lapl, min_lambda, max_lambda, increment):
    optim_lambda = optimize_lambda(N, Q_arr, y_vec, Lapl, 
                                 min_lambda, max_lambda, increment)
    Q_arr = Q_arr.as_matrix()
    N = Q_arr.shape[1]
    y_vec_arr = y_vec.as_matrix().reshape(len(y_vec),1)
    optim_f_vec = solve_f(Q_arr, Lapl, y_vec_arr, optim_lambda)
    return optim_f_vec
    
def main(N, Q_arr, y_vec, Lapl, min_lambda, max_lambda, increment, optim, 
         Q_test_arr, y_test_vec_arr):
    optim_f_vec = build_model(N, Q_arr, y_vec, Lapl, 
                              min_lambda, max_lambda, increment)
    pred = predict_travel_time(optim_f_vec, Q_test_arr)
    diff = validate(y_test_vec_arr, pred)
    return diff

if __name__ == "__main__":
    pass