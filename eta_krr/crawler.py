#==============================================================================
# writes new file with freq, file and path (node seq) for all 324 trips
#==============================================================================

import os
import  pickle

datfile = os.path.join(r"/media/asifrehan/REHAN'S/AR_personal/val_dataset",
             "val_dataset_output/sys.stdout.txt")
wrtfile = os.path.join(r"/media/asifrehan/shared_data_folder",
                        "Google Drive/UConn MS/Py Codes/ETA_KRR/_files",
                        "trip_data.txt")
with open(datfile, "r") as srce:
    dat = srce.read().replace("\n", "")
    tgt = open(wrtfile, "w")
    start = 0
    
    for i in range(324): 
        path_start = dat.find("[", start)
        path_end = dat.find("]", path_start)
        meta_start = path_end + 2
        meta_end = dat.find("csv", meta_start) + 3
        path = dat[path_start : path_end+1]
        meta = dat[meta_start : meta_end]
        tgt.write(meta+"\n"+path)
        start = meta_end
        


#==============================================================================
#  
#==============================================================================
trip_road_id_file = os.path.join(r"/media/asifrehan/shared_data_folder",
                        "Google Drive/UConn MS/Py Codes/ETA_KRR/_files",
                        "trip_road_id.txt")

mg = os.path.join('/media/asifrehan/shared_data_folder/Google Drive',
                      'UConn MS/Py Codes/ETA_KRR/_files/MultiGraph.p')
mg = pickle.load(open(mg, 'rb'))
with open(wrtfile, "r") as nodeseq:
    dat = nodeseq.read().replace("\n", "")
    tgt = open(trip_road_id_file, "w")
    start = 0
    for i in range(324):
        meta_start = dat.find('Freq', start)
        meta_end = dat.find("csv", meta_start) + 3
        meta = dat[meta_start : meta_end]
        nodes_list_start = dat.find('[', meta_end)
        nodes_list_end = dat.find(']', nodes_list_start)
        nodes_list_str = dat[nodes_list_start:nodes_list_end+1]
        nodes_list_fl = []
        nodes_par_opn = 0
        nodes_par_cls = 0
        while nodes_par_cls+1 <= len(nodes_list_str)-1:
            nodes_par_opn = nodes_list_str.find('(', nodes_par_opn)
            nodes_tup_comma = nodes_list_str.find(',', nodes_par_opn)
            node_x = float(nodes_list_str[nodes_par_opn+1 : nodes_tup_comma])
            nodes_par_cls = nodes_list_str.find(')', nodes_tup_comma)
            node_y = float(nodes_list_str[nodes_tup_comma+2 : nodes_par_cls])
            node_fl = (node_x, node_y)
            nodes_list_fl.append(node_fl)
            nodes_par_opn = nodes_par_cls 
        
        road_id_seq = []
        
        
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                try :
                    road_id_seq.append(mg[nodes[i]][nodes[j]]).keys()
                except:
                    continue
        tgt.write(meta+"\n"+str(road_id_seq))
        start = nodes_list_end+1
                
                