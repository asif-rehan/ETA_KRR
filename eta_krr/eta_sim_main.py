'''
Created on Jun 6, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import pandas as pd
from scipy import stats
import pickle 
import networkx as nx
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
M = 177
onboard_time_min = 2
onboard_time_max = 15
overlap_max_minute = 15
overlap_dir = 1    #or -1 for sparse
lamb_min = 1
lamb_max= 5000
lamb_step = 1
seg = [(TOD, DOW) for TOD in ['af', 'ev', 'mo'] for 
    DOW in ['thu', 'tue', 'wed']]

def crowd_density(train_len_indic_mat, 
                  LinkIdLenSer=os.path.join(r'C:\Users\asr13006\Google Drive',
                    r'UConn MS\Py Codes\HMM_Krumm_Newson_Implementation',
                    r'MM_AR\Relevant_files\LinkIdLenSer.p')):
    id_len_ser = pickle.load(open(LinkIdLenSer, 'rb'))
    
    N= train_len_indic_mat.shape[1]
    M= train_len_indic_mat.shape[0]
    link_crowd_density_series =   \
                        train_len_indic_mat.mean(axis=1)/id_len_ser
    network_crowd_density = link_crowd_density_series.sum(axis=0) / float(M)
    return link_crowd_density_series, network_crowd_density

def inner_loop(dow, tod, corr_coef, onboard_time_max, overlap_dir, 
               overlap_max_minute):
    train_len_indic_mat,train_experienced_time = crowd_source_simu(
                                                    rd_files_df,
                                                    src_fldr,
                                                    tod, dow, 
                                                    M, 
                                                    onboard_time_min, 
                                                    onboard_time_max, 
                                                    overlap_max_minute, 
                                                    overlap_dir)
    
    lcd, ncd = crowd_density(train_len_indic_mat) 
    
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
                                                M, 
                                                onboard_time_min, 
                                                onboard_time_max, 
                                                overlap_max_minute, 
                                                1)
    test_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                      1.0/speed_vec_df, 
                                        test_link_indic_mat.as_matrix())
    #val_link_indic_mat = pickle.load(open(dow+'_'+tod+'_len_indic_mat.p',
    #                                      'rb'))
    #val_experiece_tim = pickle.load(open(dow+'_'+tod+'_hop_time.p','rb'))    
    
    #val_pred_experience_time = process.predict_travel_time(optim_f_vec, 
    #                                                  1.0/speed_vec_df, 
    #                                        val_link_indic_mat.as_matrix())
    #fig = plt.figure()
    #ax = plt.axes()
    #ax.plot(test_experience_time, test_pred_experience_time, 'o')
    cor_r, two_tail_p_value = stats.pearsonr(
                                        test_experience_time.as_matrix(), 
                                    test_pred_experience_time.flatten())
    diff = test_pred_experience_time.flatten()-test_experience_time.as_matrix()
    pct_diff = (test_pred_experience_time.flatten() - 
                test_experience_time.as_matrix())/  \
                test_experience_time.as_matrix()*100
                
    corr_coef.append((overlap_dir, dow, tod, cor_r, 
                      lcd, ncd, onboard_time_max,
                      opt_lambda, diff.min(), diff.max(), diff.mean(), 
                      pct_diff.min(), pct_diff.max(), pct_diff.mean()))
    print corr_coef[-1]
    return optim_f_vec, opt_lambda, cor_r, corr_coef


def disagg(seg, repeat):    
    corr_coef = []
    for tod, dow in seg:
        for obd_max in  [5]:#[15, 10]:
            for overlap_dir in [1, -1]:
                for i in range(repeat):
                    overlap_max_minute = obd_max
                    f_vec, lam, r, cor_coef = inner_loop(dow, tod, corr_coef, 
                                                         obd_max, overlap_dir,
                                                         overlap_max_minute)
                                
    summary= pd.DataFrame(corr_coef, columns=['Overlap', 'DOW', 'TOD', 
                                       'Corr Coef', 
                                       'LCD', 'NCD', 'max onboard time (sec)',
                                       'Lambda*', 'min exp time diff(sec)',
                                       'max exp time diff(sec)', 
                                       'mean exp time diff(sec)',
                                       'Min exp time % diff', 
                                       'Max exp time % diff',
                                       'Mean exp time % diff'])
    
    return summary
    
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
    return None

disagg(seg[:3], 5).to_csv(r'../_files/eta_krr_plots/disagg_summary_af_5.csv')
disagg(seg[3:6], 5).to_csv(r'../_files/eta_krr_plots/disagg_summary_ev_5.csv')
disagg(seg[6:9], 5).to_csv(r'../_files/eta_krr_plots/disagg_summary_mo_5.csv')
