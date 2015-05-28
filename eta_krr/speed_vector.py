'''
Created on May 27, 2015

@author: asr13006
'''
import os
import pandas as pd
import numpy as np
import pickle
import dateutil.parser as parser
#----------------------------------------------------------------------------- 
src_fldr = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS',
                    r'Py Codes\ETA_KRR\_files\files_for_ETA_simulation')
node_files = [f for f in os.listdir(os.path.join(src_fldr,'node_files'))  \
             if os.path.isfile(os.path.join(src_fldr,'node_files',f))]
road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files'))  \
             if os.path.isfile(os.path.join(src_fldr,'road_files',f))]
nd_rd_pair_files = zip(node_files, road_files)
n_road = 177
#----------------------------------------------------------------------------- 

speed_storage = {}
for i in xrange(n_road):
    speed_storage[i] = []
speed_vector = np.zeros((1,n_road))

for (v, e) in nd_rd_pair_files:
    node_f = pd.read_csv(os.path.join(src_fldr, 'node_files', v),
                         index_col=0,
                         usecols = [0,3,4])
    road_f = pd.read_csv(os.path.join(src_fldr, 'road_files', e),index_col=0)
    road_f['speed'] = ""
    ts_idx = node_f[node_f.timestamp != '0'].index
    delta_ts_ticks = zip(ts_idx[:-1], ts_idx[1:])
    for (t_prev, t_nxt) in delta_ts_ticks:
        includd_0_spd = road_f[t_prev:t_nxt]
        excludd_0_spd = includd_0_spd[includd_0_spd.length != 0]
        if len(excludd_0_spd.index) != 0:
            t_delta = parser.parse(node_f.timestamp[t_nxt]) -   \
                                    parser.parse(node_f.timestamp[t_prev])
            avg_speed = np.divide(excludd_0_spd.length.sum(), 
                              t_delta.total_seconds()) 
            for idx in excludd_0_spd.index:
                road_f.speed[idx] = avg_speed
    for idx in road_f[road_f.speed!=''].index:
        speed_storage[road_f.road_id[idx]].append(road_f.speed[idx])
pickle.dump(speed_storage, open('speed_storage.p','wb'))
for i in xrange(n_road):
    speed_vector[0][i] = np.mean(np.array(speed_storage[i]))
pickle.dump(speed_storage, open('speed_vector.p','wb'))    