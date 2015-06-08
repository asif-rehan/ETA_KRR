'''
Created on Jun 5, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import numpy as np
from scipy import linalg

def solve_f(Q_arr, Lapl, y_vec_arr, reg_lambda):
    A = Q_arr.dot(Q_arr.T) + reg_lambda*Lapl
    b = Q_arr.dot(y_vec_arr)
    f_vec = linalg.solve(A, b) 
    return f_vec

def optimize_lambda(N, Q_arr, y_vec_arr, Lapl, 
                    min_lambda, max_lambda, increment, fast=True):
    error_threshold = np.inf
    error_log = []
    for lambda_now in np.arange(min_lambda, max_lambda, increment):
        if fast:
            try:
                error = fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
            except:
                error = slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
        else:
            error = slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, lambda_now)
        error_log.append((lambda_now, error))
        if error < error_threshold:
            LOOCV_argmin_lambda = lambda_now
            error_threshold = error
    return LOOCV_argmin_lambda, error_log 

def fast_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    I = np.identity(N)
    inverted = linalg.inv(Q_arr.dot(Q_arr.T) + reg_lambda*Lapl)
    H = Q_arr.T.dot(inverted.dot(Q_arr))
    I_minus_H = I - H
    mat= linalg.inv(linalg.block_diag(I_minus_H)).dot(I_minus_H.dot(y_vec_arr))
    LOOCV_cost = mat.T.dot(mat)[0][0]
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost

def slow_LOOCV_cost(N, Q_arr, y_vec_arr, Lapl, reg_lambda):
    LOOCV_cost = 0
    for n in xrange(N): 
        _Q_leave_n = np.delete(Q_arr, n, 0)
        _y_leave_n = np.delete(y_vec_arr, n, 0)
        _f_vec_leave_n = solve_f(_Q_leave_n, Lapl, _y_leave_n, reg_lambda)
        LOOCV_cost += (y_vec_arr[n] - Q_arr[:, n].T.dot(_f_vec_leave_n))^2
    avg_LOOCV_cost = LOOCV_cost/float(N)
    return avg_LOOCV_cost 

def predict_travel_time(optim_f_vec, speed_vec_arr, Q_test_arr):
    """speed_vec_arr : vector of inverse avg speed"""
    state_inv_speed = optim_f_vec + speed_vec_arr
    return Q_test_arr.T.dot(state_inv_speed)

def validate(y_test_vec_arr, pred_y_vec_arr):
    diff = pred_y_vec_arr - y_test_vec_arr
    return diff

def build_model(Q_df, y_vec_df, speed_vec_arr, Lapl, 
                min_lambda, max_lambda, increment):
    """
    y_vec_df : onbrd_experienced_time vector in pandas.DF
    y_dev_arr : vector after subtracting avg_onboard_experience_time
            avg_onboard_experience_time = sum(links involved/avg link speed)
    """
    Q_arr = Q_df.as_matrix()
    N = len(y_vec_df)
    assert Q_arr.shape[1] == y_vec_df.shape[0]
    assert Q_arr.shape[0] == speed_vec_arr.shape[0]
    y_vec_arr = y_vec_df.as_matrix().reshape(N,1)
    
    inv_speed_vec = 1.0 / speed_vec_arr
    y_dev_vec_arr = y_vec_arr - Q_arr.T.dot(inv_speed_vec)
    
    optim_lambda, error_log = optimize_lambda(N, Q_arr, y_dev_vec_arr, Lapl, 
                                 min_lambda, max_lambda, increment)
    
    optim_f_vec = solve_f(Q_arr, Lapl, y_dev_vec_arr, optim_lambda)
    return optim_f_vec, optim_lambda, error_log
    
def main(Q_arr, y_vec_df, speed_vec_files_df, Lapl, 
         min_lambda, max_lambda, increment, optim, 
         Q_test_arr, y_test_vec_arr):
    optim_f_vec = build_model(Q_arr, y_vec_df, speed_vec_files_df, Lapl, 
                              min_lambda, max_lambda, increment)[0]
    speed_vec_arr = 1.0/ speed_vec_files_df
    pred = predict_travel_time(optim_f_vec, speed_vec_arr, Q_test_arr)
    diff = validate(y_test_vec_arr, pred)
    return diff

if __name__ == "__main__":
    main()