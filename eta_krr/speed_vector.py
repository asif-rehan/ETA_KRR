'''
Created on May 27, 2015

@author: Asif.Rehan@engineer.uconn.edu
'''
import os
import pandas as pd
import numpy as np
import pickle
from dateutil import parser

def speed_vector(src_fldr, nd_rd_pair_files, n_road):
    """post-processing tool following map-matching.
    Works on map-matched node and road output
    Calculates speed for each road when vehicle was not static
    Adds three new columns to the road output dataframe
    
    Speed:  Measured by the previous and next timestamp available and 
            the distance covered within that time difference. 
            Ignores static ones
    ts_idx_prev: Timestamp index from node file for immediate before as 
                    vehicle is on or about to move
    ts_idx_next: Timestamp index immediately after as vehicle is on move or 
                    about to stop 
    ts_prev: Timestamp immediately before as vehicle is on or about to move
    ts_next: Timestamp immediately after as vehicle is on move or about to stop 
    """
    speed_storage = {}
    for i in xrange(n_road):
        speed_storage[i] = []
    
    speed_vector = np.zeros((1, n_road))
    for v, e in nd_rd_pair_files[2:]:
        node_f = pd.read_csv(os.path.join(src_fldr, 'node_files', v), 
            index_col=0, 
            usecols=[0, 3, 4])
        road_f = pd.read_csv(os.path.join(src_fldr, 'road_files', e), 
                             index_col=0)
        road_f['speed_mps']       = ""
        road_f['ts_delta_sec']    = ""
        road_f['ts_idx_prev'] = ""
        road_f['ts_idx_next'] = ""
        road_f['ts_prev']     = ""
        road_f['ts_next']     = ""

        
        ts_idx = node_f[node_f.timestamp != '0'].index
        delta_ts_ticks = zip(ts_idx[:-1], ts_idx[1:])
        for ts_idx_prev, ts_idx_next in delta_ts_ticks:
            includd_0_spd = road_f[ts_idx_prev:ts_idx_next]
            excludd_0_spd = includd_0_spd[includd_0_spd.length != 0]
            if len(excludd_0_spd.index) != 0:
                ts_next = parser.parse(node_f.timestamp[ts_idx_next])
                ts_prev = parser.parse(node_f.timestamp[ts_idx_prev])
                ts_delta = ts_next - ts_prev
                avg_speed = np.divide(excludd_0_spd.length.sum(), 
                    ts_delta.total_seconds())
                for idx in excludd_0_spd.index:
                    road_f.set_value(idx, 'speed_mps', avg_speed)
                    road_f.set_value(idx, 'ts_delta_sec', 
                                                    ts_delta.total_seconds())
                    road_f.set_value(idx, 'ts_idx_prev', ts_idx_prev)
                    road_f.set_value(idx, 'ts_idx_next', ts_idx_next)
                    road_f.set_value(idx, 'ts_prev', ts_prev)
                    road_f.set_value(idx, 'ts_next', ts_next)

        road_f.to_csv(os.path.join(src_fldr, 'road_files', e))
        for idx in road_f[road_f.speed_mps != ''].index:
            speed_storage[road_f.road_id[idx]].append(road_f.speed_mps[idx])
    for i in xrange(n_road):
        speed_vector[0][i] = np.mean(np.array(speed_storage[i]))
    return speed_vector, speed_storage

if __name__ == "__main__":
    src_fldr = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS',
                        r'Py Codes\ETA_KRR\_files\files_for_ETA_simulation')
    node_files = [f for f in os.listdir(os.path.join(src_fldr,'node_files'))  \
                 if os.path.isfile(os.path.join(src_fldr,'node_files',f))]
    road_files = [f for f in os.listdir(os.path.join(src_fldr,'road_files'))  \
                 if os.path.isfile(os.path.join(src_fldr,'road_files',f))]
    nd_rd_pair_files = zip(node_files, road_files)
    n_road = 177 
    speed_vec, speed_stor = speed_vector(src_fldr, nd_rd_pair_files, n_road)
    pickle.dump(speed_stor, open('speed_storage.p', 'wb'))    
    pickle.dump(speed_vec, open('speed_vector.p', 'wb'))