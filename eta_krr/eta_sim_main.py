'''
Created on Jun 6, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import pandas as pd
from scipy import stats
import pickle 
import process
from simulation_sampler import crowd_source_simu
import matplotlib.pyplot as plt
 
this_dir =  os.path.dirname(__file__)
src_fldr = os.path.join(this_dir, r'../_files/files_for_ETA_simulation')
road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files')) 
              if os.path.isfile(os.path.join(src_fldr, 'road_files', f))]
store = [(f[12:14], f[15:18], f[19:21], f) for f in road_files]
rd_files_df = pd.DataFrame(store, 
                           columns=['route', 'DOW', 'TOD', 'road_file'])
speed_vec_files = [f for f in os.listdir(os.getcwd()) 
                   if os.path.isfile(f) and f[-14:-2] == 'speed_vector']
speed_vec_store = [(f[:3], f[4:6], f) for f in speed_vec_files]
speed_vec_files_df = pd.DataFrame(speed_vec_store, 
                            columns=['DOW', 'TOD', 'speed_vec_file'])
Lapl = pickle.load(open("Laplacian_matrix.p", 'rb'))
n_road = 177
onboard_time_min = 2
onboard_time_max = 15
overlap_max_minute = 15
overlap_dir = 1 #or -1 indicates how time-separate CS
lamb_min = 0.1
lamb_max= 10000
lamb_step = 1
seg = [(TOD, DOW) for TOD in ['af', 'ev', 'mo'] for 
    DOW in ['thu', 'tue', 'wed']]
corr_coef = []
for tod, dow in seg:
    train_len_indic_mat,train_experienced_time = crowd_source_simu(rd_files_df,
                                                src_fldr,
                                                tod, dow, 
                                                n_road, 
                                                onboard_time_min, 
                                                onboard_time_max, 
                                                overlap_max_minute, 
                                                overlap_dir)
    speed_vec_file = speed_vec_files_df.loc[
                                        (speed_vec_files_df['DOW']=='thu') & 
                                        (speed_vec_files_df['TOD']=='mo'), 
                                        'speed_vec_file'].values[0]
    speed_vec_df = pickle.load(open(speed_vec_file, 'rb'))
    optim_f_vec, opt_lambda, err_log = process.build_model(train_len_indic_mat, 
                                      train_experienced_time,
                                      speed_vec_df, 
                                      Lapl, lamb_min, lamb_max, lamb_step)
    test_len_indic_mat, test_experience_time = crowd_source_simu(rd_files_df,
                                                src_fldr,
                                                tod, dow, 
                                                n_road, 
                                                onboard_time_min, 
                                                onboard_time_max, 
                                                overlap_max_minute, 
                                                overlap_dir)
    pred_experiece_time = process.predict_travel_time(optim_f_vec, 
                                                      1.0/speed_vec_df, 
                                                test_len_indic_mat.as_matrix())
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(test_experience_time, pred_experiece_time, 'o')
    cor = stats.pearsonr(test_experience_time.as_matrix(), 
                         pred_experiece_time.flatten())
    corr_coef.append((dow, tod, cor))
    ttl = 'Predicted versus Actual Time - {0} {1}'.format(dow, tod)
    plt.title(ttl)
    plt.xlabel('Actual (sec)')
    plt.ylabel('Predicted (sec)')
    plt.savefig('../_files/eta_krr_plots/{0}'.format(ttl))
    plt.close()
    
    fig = plt.figure()
    ax = plt.axes()
    errors, lambda_values = zip(*err_log)
    plt.plot(lambda_values, errors)
    plt.xscale('log') 
    plt.yscale('log')
    ttl = 'LOOCV Error versus Lambda {0} {1}'.format(dow, tod)
    plt.title(ttl)
    plt.xlabel('log(lambda)')
    plt.ylabel('log(LOOCV Error)')
    plt.savefig('../_files/eta_krr_plots/{0}'.format(ttl))
    plt.close()

print pd.DataFrame(corr_coef, columns=['DOW', 'TOD', 'Corr Coef'])