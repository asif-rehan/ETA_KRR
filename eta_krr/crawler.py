import os
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