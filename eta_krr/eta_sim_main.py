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
import matplotlib as mpl
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
LinkIdLenSer=os.path.join(r'C:\Users\asr13006\Google Drive',
                    r'UConn MS\Py Codes\HMM_Krumm_Newson_Implementation',
                    r'MM_AR\Relevant_files\LinkIdLenSer.p')
id_len_series = pkl.load(open(LinkIdLenSer,'rb'))
link_len_vec = id_len_series.as_matrix().reshape(len(id_len_series),1)
Lapl = pkl.load(open("Laplacian_matrix.p", 'rb'))

M = 177
onboard_time_min = 2
onboard_time_max = 15
overlap_max_minute = 15
overlap_dir = 1    #or -1 for sparse
lamb_min = 1
lamb_max= 5000
lamb_step = 1

def crowd_density(train_len_indic_mat, link_len_vec):
    """Measures redundancy measures of the crowd-sourced data
    
    Returns
    ------- 
    Redundancy : (How many times the link was observed - 1)
                Indicates excess information than needed to have it once
                If not observed at all, it is -1 for that link
    Redundant Length Present : (Total length covered in k traces - k*length)
                Assuming link was observed in k-traces
                Measures its redundant presence influencing least square calc  
    """
    M= train_len_indic_mat.shape[0]
    mat = train_len_indic_mat.as_matrix()
    cover = mat - link_len_vec  
    count_redundancy = np.zeros((M,1)) 
    length_redundacy = np.zeros((M,1))
    for i in range(M):
        row = cover[i]
        count_redundancy[i] = len(row[row >= 0]) - 1
        if len(row[row >= 0]) > 0:
            length_redundacy[i] = row[row >= 0].sum()/link_len_vec[i]
        else:
            length_redundacy[i] = -1
    return count_redundancy, length_redundacy


def get_metrics(test_pred_experience_time, test_experience_time,
                dow, tod, onboard_time_max, overlap_max_minute, 
                speed_vec_df, optim_f_vec):
    test_rmse = process.calc_rmse(test_pred_experience_time, 
                                  test_experience_time.as_matrix())
    test_exp_arr = test_experience_time.as_matrix()
    test_coeff_corr, test_p_value, test_se_est = stats.linregress(
                                           test_pred_experience_time.flatten(),
                                           test_exp_arr)[2:]
    test_coeff_det = test_coeff_corr**2
    diff = test_pred_experience_time.flatten()-test_experience_time.as_matrix()
    pct_diff = (test_pred_experience_time.flatten() - 
                test_experience_time.as_matrix())/  \
                test_experience_time.as_matrix()*100                                        
    metrics =  test_rmse, test_coeff_corr, test_coeff_det, \
                test_p_value, test_se_est,  \
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
    speed_vec_file = speed_vec_files_df.loc[
                                (speed_vec_files_df['DOW']==speed_vec_dow) & 
                                (speed_vec_files_df['TOD']==speed_vec_tod), 
                                'speed_vec_file'].values[0]
    speed_vec_arr = pkl.load(open(speed_vec_file, 'rb'))
    optim_f_vec, opt_lambda, err_log =process.build_model(train_link_indic_mat, 
                                      train_experienced_time,
                                      speed_vec_arr, 
                                      Lapl, lamb_min, lamb_max, lamb_step)
    #==========================================================================
    # make LOO CV plot here
    #==========================================================================
    #==========================================================================
    # make heatmap and scatterplot on train dataset
    #==========================================================================
    train_pred_experience_time = process.predict_travel_time(optim_f_vec, 
                                                1.0 / speed_vec_arr, 
                                            train_link_indic_mat.as_matrix())
    train_metrics = get_metrics(train_pred_experience_time, 
                                train_experienced_time, 
                                dow, tod, onboard_time_max, overlap_max_minute, 
                                speed_vec_arr, optim_f_vec)
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
                                                        1.0 / speed_vec_arr, 
                                            test_link_indic_mat.as_matrix())
    test_metrics = get_metrics(test_pred_experience_time, test_experience_time,
                               dow, tod, onboard_time_max, overlap_max_minute, 
                               speed_vec_arr, optim_f_vec)
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
                                                        1.0 / speed_vec_arr, 
                                            val_link_indic_mat.as_matrix())
        val_metrics = get_metrics(val_pred_experience_time, val_experiece_time,
                                  dow, val_tod, onboard_time_max, 
                                  overlap_max_minute, speed_vec_arr, 
                                  optim_f_vec)
        val_metrics_list.append((val_tod, val_metrics))
    #==========================================================================
    # Redundancy 
    #==========================================================================
    count_redunt, len_redunt = crowd_density(train_link_indic_mat,link_len_vec) 
    
    tt_change =  (link_len_vec * optim_f_vec).flatten()
    cnt_rdnt_R, cnt_rdnt_p_value = stats.linregress(count_redunt.flatten(),
                                                        tt_change)[2:4]
    len_rdnt_R, len_rdnt_p_value = stats.linregress(len_redunt.flatten(),
                                                        tt_change)[2:4]
    #==========================================================================
    #congestion_heatmap(dow, tod, val_tod, overlap_dir, tt_change, 
    #                   count_redunt.flatten(), len_redunt.flatten())
    
    return opt_lambda, cnt_rdnt_R**2, cnt_rdnt_p_value,  \
            len_rdnt_R**2, len_rdnt_p_value, train_metrics, \
            test_metrics, val_metrics_list,  \
            test_experience_time.as_matrix(),  \
            test_pred_experience_time.flatten(),  \
            train_pred_experience_time.flatten(),  \
            train_experienced_time.as_matrix().flatten()
            


def run_full_output(seg, max_onboard_time_conditions=[15,10,5], 
                    speed_vec_dow='all',speed_vec_tod='af', 
                    val_tods=['mo', 'ev'],repeat=1):    
    columns=['Model_ID', 'Dataset','TOD', 'DOW',
                'Lambda', 'Max_OnBoardTime_minute', 'Sparsity', 
                'Count_Redunt_Rsq', 'Count_Redunt_pValue',
                'Length_Redunt_Rsq', 'Length_Redunt_pValue',
                'RMSE', 'Pearson_r', 'R_Squared', 
                'Slope_p_Value', 'StdErr_Slope', 
                'Min_Diff_sec', 'Max_Diff_sec','Mean_Diff_sec', 'Min_Diff_pct', 
                'Max_Diff_pct','Mean_Diff_pct']
    output_df = pd.DataFrame(columns = columns)
    model_id = 0
    for tod, dow in seg:
        scat_plt_data = []
        for obd_max in  max_onboard_time_conditions:
            for overlap_dir in [1, -1]:
                model_id += 1
                overlap_max_minute = obd_max
                out = inner_loop(dow, tod, obd_max, overlap_dir, val_tods,
                                                     overlap_max_minute,
                                                     speed_vec_dow, 
                                                     speed_vec_tod)
                opt_lam             = out[0]
                cnt_rdnt_Rsq        = out[1]
                cnt_rdnt_p_value    = out[2]
                len_rdnt_Rsq        = out[3]
                len_rdnt_p_value    = out[4]
                train_metrics       = out[5]  
                test_metrics        = out[6]
                val_metrics_list    = out[7]
                sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
                for (metrics, dat) in [(train_metrics, 'Train'),
                                   (test_metrics, 'Test')]:
                    row = [model_id, dat, tod, dow, opt_lam, 
                           obd_max, sparsity(overlap_dir), 
                           cnt_rdnt_Rsq, cnt_rdnt_p_value, 
                           len_rdnt_Rsq, len_rdnt_p_value] + list(metrics)
                    row_df = pd.DataFrame([row], columns=columns)
                    output_df = output_df.append(row_df, ignore_index=True)
                for (val_tod, metrics) in val_metrics_list:
                    row = [model_id,'Validation', val_tod,dow, 
                           opt_lam,obd_max, sparsity(overlap_dir), 
                           cnt_rdnt_Rsq, cnt_rdnt_p_value, 
                           len_rdnt_Rsq, len_rdnt_p_value] + list(metrics)
                    row_df = pd.DataFrame([row], columns=columns)
                    output_df = output_df.append(row_df, ignore_index=True)
                scat_plt_data.append((obd_max,sparsity(overlap_dir),
                                      out[8:10], out[10:12]))
        scatter_plots(dow, tod,scat_plt_data)
    return output_df

def congestion_heatmap(dow, tod, val_tod, overlap_dir_tag, 
                       tt_change, count_redunt, len_redunt):
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.set_size_inches(7, 10, forward=True)
    mpl.rcParams.update({'font.size': 8})
    fig.suptitle('Data Redundancy Vs Predicted Travel Time Deviation',
                 fontsize=14)
    axes[0].set_title('Link Travel Time Deviation')
    axes[0].set_axis_bgcolor('k')
    mm_val('').plot_roadnetwork(axes[0], fig, select=False, heatmap=True, 
                                heatmap_cmap=tt_change)
    axes[1].set_title('Link Count Redundancy')
    axes[1].set_axis_bgcolor('k')
    mm_val('').plot_roadnetwork(axes[1], fig, select=False, heatmap=True, 
                heatmap_cmap=count_redunt, heat_label='Link Count Redundancy')
    axes[2].set_title('Link Length Redundancy')
    axes[2].set_axis_bgcolor('k')
    mm_val('').plot_roadnetwork(axes[2], fig, select=False, heatmap=True, 
                heatmap_cmap=len_redunt,heat_label='Link Length Redundancy')
    
    fig.text(0.5, 0.04, 'Easting', ha='center', va='center')
    fig.text(0.05, 0.5, 'Northing', ha='center', va='center', 
             rotation='vertical')

    sparsity = lambda overlap_dir: 'Sparse' if overlap_dir==-1  \
                                                        else 'Continuous'
    fig.savefig('../_files/eta_krr_plots/{2}_{0}_{1}'.format(dow.upper(), 
                                                             tod.upper(), 
                                                    sparsity(overlap_dir_tag)))
    fig.text(0.8, 0.05, sparsity(overlap_dir_tag)+'_'+tod+'_'+tod, color='red', 
        bbox=dict(facecolor='none', edgecolor='red'))    
    plt.close()
    return None

def scatter_plots(dow, tod, scat_plt_data):
    fig, axes = plt.subplots(nrows=len(scat_plt_data)/2, 
                             ncols=2, sharex=True, sharey=True)
    fig.set_size_inches(8, 10, forward=True)
    mpl.rcParams.update({'font.size': 8})
    fig.suptitle('Predicted vs Actual Travel Time', fontsize=14)
    for i in range(len(scat_plt_data)):
        row = i//2
        col = i%2
        axes[row,col].scatter(scat_plt_data[i][2][0],
                              scat_plt_data[i][2][1], label='Test Data',
                              s=8, c='r', marker='+', alpha=0.25)
        axes[row,col].scatter(scat_plt_data[i][3][0],
                              scat_plt_data[i][3][1], label='Train Data',
                              s=8, c='b', marker='x', alpha=0.75)
        if col==1:
            axes[row,col].yaxis.set_label_position("right")
            axes[row,col].set_ylabel(str(scat_plt_data[i][0])+'Minutes', 
                                 rotation='vertical')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    fig.text(0.50, 0.04, 'Actual Time', ha='center', va='center')
    fig.text(0.33, 0.05, scat_plt_data[0][1], ha='center', va='center')
    fig.text(0.67, 0.05, scat_plt_data[1][1], ha='center', va='center')
    fig.text(0.05, 0.5, 'Predicted Time', ha='center', va='center', 
             rotation='vertical')
    fig.text(0.95, 0.5, 'Maximum On-board Time', ha='center', va='center', 
             rotation='vertical')
    fig.savefig('../_files/eta_krr_plots/scat_{0}_{1}'.format(dow.upper(), 
                                                             tod.upper()))
    plt.tight_layout()
    plt.close()
    return None

def plotting(dow, tod, test_experience_time, test_pred_experience_time,
             opt_lambda, overlap_dir_tag, corr_coef, err_log, 
             speed_vec_df, optim_f_vec):
    fig, ax, ttl = scatter_plots(dow, tod, test_experience_time, 
                                 test_pred_experience_time, opt_lambda, overlap_dir_tag, corr_coef)
    
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
    
    congestion_heatmap(dow, tod, overlap_dir_tag, speed_vec_df, 
                       optim_f_vec, fig, ax)
    return None

#disagg(seg[0], 1).to_csv(r'../_files/eta_krr_plots/disagg_summary_all.csv')
#==============================================================================
# for all-dow, afternoon 
#==============================================================================

if __name__ == '__main__':
    seg = [(TOD, DOW) for TOD in ['af'] for 
                        DOW in ['thu']]
    allout= run_full_output(seg, max_onboard_time_conditions=[15, 10], #5],
                                val_tods=['mo'])#, 'ev'])
    allout.to_csv('../_files/eta_krr_plots/ALLOUTPUT.csv')