'''
Created on Jun 6, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import numpy as np
import pandas as pd
from scipy import stats, linalg
import pickle as pkl
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

Lapl = pkl.load(open("Laplacian_matrix.p", 'rb'))
M = 177
onboard_time_min = 2
onboard_time_max = 15
overlap_max_minute = 15
overlap_dir = 1    #or -1 for sparse
lamb_min = 1
lamb_max= 5000
lamb_step = 1

def crowd_density(train_len_indic_mat, 
                  LinkIdLenSer=os.path.join(r'C:\Users\asr13006\Google Drive',
                    r'UConn MS\Py Codes\HMM_Krumm_Newson_Implementation',
                    r'MM_AR\Relevant_files\LinkIdLenSer.p')):
    id_len_ser = pkl.load(open(LinkIdLenSer, 'rb'))
    
    N= train_len_indic_mat.shape[1]
    M= train_len_indic_mat.shape[0]
    link_crowd_density_series =   \
                        train_len_indic_mat.mean(axis=1)/id_len_ser
    network_crowd_density = link_crowd_density_series.sum(axis=0) / float(M)
    return link_crowd_density_series, network_crowd_density


def get_metrics(test_pred_experience_time, test_experience_time,
                dow, tod, onboard_time_max, overlap_max_minute, 
                speed_vec_df, optim_f_vec):
    test_rmse = process.calc_rmse(test_pred_experience_time, 
                                  test_experience_time.as_matrix())
    test_exp_arr = test_experience_time.as_matrix()
    test_Rsq, test_p_value, test_se = stats.linregress(
                                           test_pred_experience_time.flatten(),
                                           test_exp_arr)[2:]
    diff = test_pred_experience_time.flatten()-test_experience_time.as_matrix()
    pct_diff = (test_pred_experience_time.flatten() - 
                test_experience_time.as_matrix())/  \
                test_experience_time.as_matrix()*100                                        
    metrics =  test_rmse, test_Rsq**2, test_p_value, test_se,  \
               diff.min(), diff.max(), diff.mean(),  \
                pct_diff.min(), pct_diff.max(), pct_diff.mean()
    return metrics

def inner_loop(dow, tod, onboard_time_max, overlap_dir, val_tods,
               overlap_max_minute, speed_vec_dow, speed_vec_tod):
    train_link_indic_mat,train_experienced_time = crowd_source_simu(
                                                    rd_files_df,
                                                    src_fldr,
                                                    tod, dow, 
                                                    M, 
                                                    onboard_time_min, 
                                                    onboard_time_max, 
                                                    overlap_max_minute, 
                                                    overlap_dir)
    lcd, ncd = crowd_density(train_link_indic_mat) 
    
    speed_vec_file = speed_vec_files_df.loc[
                                (speed_vec_files_df['DOW']==speed_vec_dow) & 
                                (speed_vec_files_df['TOD']==speed_vec_tod), 
                                'speed_vec_file'].values[0]
    speed_vec_df = pkl.load(open(speed_vec_file, 'rb'))
    optim_f_vec, opt_lambda, err_log =process.build_model(train_link_indic_mat, 
                                      train_experienced_time,
                                      speed_vec_df, 
                                      Lapl, lamb_min, lamb_max, lamb_step)
    #==========================================================================
    # make LOO CV plot here
    #==========================================================================
    #==========================================================================
    # make heatmap and scatterplot on train dataset
    #==========================================================================
    train_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                1.0 / speed_vec_df, 
                                            train_link_indic_mat.as_matrix())
    train_metrics = get_metrics(train_pred_experience_time, 
                                train_experienced_time, 
                                dow, tod, onboard_time_max, overlap_max_minute, 
                                speed_vec_df, optim_f_vec)
    #==========================================================================
    #Test set
    #==========================================================================
    test_link_indic_mat, test_experience_time = crowd_source_simu(rd_files_df, 
                                                        src_fldr, 
                                                        tod, dow, 
                                                        M, 
                                                        onboard_time_min, 
                                                        onboard_time_max, 
                                                        overlap_max_minute, 
                                                        1)
    test_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                        1.0 / speed_vec_df, 
                                            test_link_indic_mat.as_matrix())
    test_metrics = get_metrics(test_pred_experience_time, test_experience_time,
                               dow, tod, onboard_time_max, overlap_max_minute, 
                               speed_vec_df, optim_f_vec)
    #==========================================================================
    # make heatmap and scatterplot on test dataset
    #==========================================================================
    #==========================================================================
    # validation set
    #==========================================================================
    val_metrics_list= []
    for val_tod in val_tods:
        val_link_indic_mat = pkl.load(open(dow+'_'+val_tod+
                                              '_len_indic_mat.p','rb'))
        val_experiece_time = pkl.load(open(dow+'_'+val_tod+'_hop_time.p','rb')) 
        val_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                        1.0 / speed_vec_df, 
                                            val_link_indic_mat.as_matrix())
        val_metrics = get_metrics(val_pred_experience_time, val_experiece_time,
                                  dow, val_tod, onboard_time_max, 
                                  overlap_max_minute, speed_vec_df, 
                                  optim_f_vec)
        val_metrics_list.append((val_tod, val_metrics))
    #==========================================================================
    # make heatmap and scatterplot on val datasets here
    #==========================================================================
    
    #fig = plt.figure()
    #ax = plt.axes()
    #ax.plot(test_experience_time, test_pred_experience_time, 'o')
    #==========================================================================
            
    return opt_lambda, ncd, train_metrics, test_metrics, val_metrics_list


def run_full_output(seg, max_onboard_time_conditions=[15,10,5], 
                    speed_vec_dow='all',speed_vec_tod='af', 
                    val_tods=['mo', 'ev'],repeat=1):    
    columns=['Model_ID', 'Dataset','TOD', 'DOW',
                'Lambda', 'Max_OnBoardTime_minute', 'Sparsity', 
                'Network_Crowding_Density', 'RMSE', 'R_Squared', 
                'Slope_p_Value', 'StdErr', 
                'Min_Diff_sec', 'Max_Diff_sec','Mean_Diff_sec', 'Min_Diff_pct', 
                'Max_Diff_pct','Mean_Diff_pct']
    output_df = pd.DataFrame(columns = columns)
    model_id = 0
    for tod, dow in seg:
        for obd_max in  max_onboard_time_conditions:
            for overlap_dir in [1, -1]:
                model_id += 1
                
                overlap_max_minute = obd_max
                out = inner_loop(dow, tod, obd_max, overlap_dir, val_tods,
                                                     overlap_max_minute,
                                                     speed_vec_dow, 
                                                     speed_vec_tod)
                opt_lam = out[0]
                NCD = out[1]
                train_metrics = out[2]  
                test_metrics = out[3]
                val_metrics_list = out[4]
                sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
                for (metrics, dat) in [(train_metrics, 'Train'),
                                   (test_metrics, 'Test')]:
                    row = [model_id, dat, tod, dow, opt_lam, obd_max,
                           sparsity(overlap_dir), NCD] + list(metrics)
                    row_df = pd.DataFrame([row], columns=columns)
                    output_df = output_df.append(row_df, ignore_index=True)
                for (val_tod, metrics) in val_metrics_list:
                    row = [model_id,'Validation', val_tod,dow, opt_lam,obd_max,
                           sparsity(overlap_dir), NCD] + list(metrics)
                    row_df = pd.DataFrame([row], columns=columns)
                    output_df = output_df.append(row_df, ignore_index=True)
    return output_df


def congestion_heatmap(dow, tod, overlap_dir_tag, speed_vec_df, optim_f_vec, fig, ax):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_axis_bgcolor('k')
    tt_chng = speed_vec_df * optim_f_vec
    mm_val('').plot_roadnetwork(ax, fig, select=False, heatmap=True, heatmap_cmap=tt_chng.flatten())
    titl_sp_dev = 'Link Time Deviation - {2} {0} {1}'.format(
        dow.upper(), 
        tod.upper(), 
        overlap_dir_tag)
    plt.title('       ' + titl_sp_dev)
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    fig.savefig('../_files/eta_krr_plots/{}'.format(titl_sp_dev))
    plt.close()

def plotting(dow, tod, test_experience_time, test_pred_experience_time,
             opt_lambda, overlap_dir_tag, corr_coef, err_log, 
             speed_vec_df, optim_f_vec):
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
    
    congestion_heatmap(dow, tod, overlap_dir_tag, speed_vec_df, optim_f_vec, fig, ax)
    return None

#disagg(seg[0], 1).to_csv(r'../_files/eta_krr_plots/disagg_summary_all.csv')

#==============================================================================
# for all-dow, afternoon 
#==============================================================================

if __name__ == '__main__':
    seg = [(TOD, DOW) for TOD in ['af'] for 
                        DOW in ['thu']]
    print run_full_output(seg, max_onboard_time_conditions=[15],#, 10, 5],
                                val_tods=['mo', 'ev'])