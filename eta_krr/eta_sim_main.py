'''
Created on Jun 6, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import pandas as pd
from scipy import stats
import pickle 
from eta_krr import process
from eta_krr.simulation_sampler import crowd_source_simu
import matplotlib.pyplot as plt
from MM_AR_validation.validation import Validate as mm_val


this_dir =  os.path.dirname(__file__)
src_fldr = os.path.join(this_dir, r'../_files/files_for_ETA_simulation')
road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files')) 
              if os.path.isfile(os.path.join(src_fldr, 'road_files', f))]
store = [(f[12:14], f[15:18], f[19:21], f) for f in road_files]
rd_files_df = pd.DataFrame(store, 
                           columns=['route', 'DOW', 'TOD', 'road_file'])
speed_vec_files = [f for f in os.listdir(os.getcwd()) 
                   if os.path.isfile(f) and f[7:-2] == 'speed_vector']
speed_vec_store = [(f[:3], f[4:6], f) for f in speed_vec_files]
speed_vec_files_df = pd.DataFrame(speed_vec_store, 
                            columns=['DOW', 'TOD', 'speed_vec_file'])

Lapl = pickle.load(open("Laplacian_matrix.p", 'rb'))
n_road = 177
onboard_time_min = 2
onboard_time_max = 15
overlap_max_minute = 15
overlap_dir = 1    #or -1 for sparse
lamb_min = 1
lamb_max= 1000
lamb_step = 1
seg = [(TOD, DOW) for TOD in ['af', 'ev', 'mo'] for 
    DOW in ['thu', 'tue', 'wed']]

    
def results():    
    corr_coef = []
    for tod, dow in [('mo', 'tue')]:#seg:
        train_len_indic_mat,train_experienced_time = crowd_source_simu(
                                                    rd_files_df,
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
        test_link_indic_mat, test_experience_time = crowd_source_simu(rd_files_df,
                                                    src_fldr,
                                                    tod, dow, 
                                                    n_road, 
                                                    onboard_time_min, 
                                                    onboard_time_max, 
                                                    overlap_max_minute, 
                                                    1)
        test_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                          1.0/speed_vec_df, 
                                            test_link_indic_mat.as_matrix())
        val_link_indic_mat = pickle.load(open(dow+'_'+tod+'_len_indic_mat.p',
                                              'rb'))
        val_experiece_tim = pickle.load(open(dow+'_'+tod+'_hop_time.p','rb'))    
        
        val_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                          1.0/speed_vec_df, 
                                                val_link_indic_mat.as_matrix())
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(test_experience_time, test_pred_experience_time, 'o')
        cor_r, two_tail_p_value = stats.pearsonr(
                                            test_experience_time.as_matrix(), 
                                        test_pred_experience_time.flatten())
        corr_coef.append((overlap_dir_tag, dow, tod, cor_r, two_tail_p_value, 
                          opt_lambda))
        print (dow, tod, cor_r, two_tail_p_value, opt_lambda)
        ttl = 'Predicted versus Actual Time - {2} {0} {1}'.format(
                                                                dow.upper(), 
                                                                tod.upper(),
                                                            overlap_dir_tag)
        plt.title(ttl)
        plt.xlabel('Actual (sec)')
        plt.ylabel('Predicted (sec)')
        plt.savefig('../_files/eta_krr_plots/{0}'.format(ttl))
        plt.close()
        
        fig = plt.figure()
        ax = plt.axes()
        lambda_values, errors = zip(*err_log)
        plt.plot(lambda_values, errors)
        ttl = 'LOOCV Error versus Lambda - {2} {0} {1}'.format(dow.upper(), 
                                                              tod.upper(),
                                                              overlap_dir_tag)
        plt.title(ttl)
        plt.yscale('log')
        plt.xlabel('lambda')
        plt.ylabel('LOOCV Error')
        plt.axhline(min([err[1] for err in err_log]), color='r', ls='--')
        plt.axvline(opt_lambda, color='r', ls='--')
        plt.savefig('../_files/eta_krr_plots/{0}'.format(ttl))
        plt.close()
        
        fig = plt.figure()
        ax = plt.axes()
        ax.set_axis_bgcolor('k')
        tt_chng = speed_vec_df * optim_f_vec
        mm_val('').plot_roadnetwork(ax, fig, select=False, heatmap=True, 
                                    heatmap_cmap=tt_chng.flatten())
        titl_sp_dev = 'Link Time Deviation - {2} {0} {1}'.format(
                                                                dow.upper(), 
                                                                tod.upper(),
                                                            overlap_dir_tag)
        plt.title('       '+titl_sp_dev)
        
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        fig.savefig('../_files/eta_krr_plots/{}'.format(titl_sp_dev))
        plt.close()
        
    print pd.DataFrame(corr_coef, columns=['Overlap/Sparse', 'DOW', 'TOD', 
                                       'Corr Coef', 'Two-tailed p-value',
                                       'Optimized lambda'])
for overlap_dir, overlap_dir_tag in [(1, 'CONT'), (-1, 'DISC')]:
    results()