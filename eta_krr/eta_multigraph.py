import networkx as nx
import fiona, pickle
import os
#import matplotlib.pyplot as plt
from fiona.crs import from_epsg

def CreateMultiGraph(road_net_shp):
    G = nx.MultiGraph()
    with fiona.open(road_net_shp,crs= from_epsg(32618)) as shp:
        #driver='ESRI Shapefile'
        node_coord_to_id_dict = {}
        node_id = -1
        for elem in shp:
            strt = elem['geometry']['coordinates'][0]
            len_coords = len(elem['geometry']['coordinates'])
            end = elem['geometry']['coordinates'][len_coords-1]
            for node in [strt, end]:
                if node not in node_coord_to_id_dict:
                    node_coord_to_id_dict[node] = str(node_id)
                    node_id -= 1
            length_meter = elem['properties']['Length_met']
            G.add_edge(strt, end, 
                       len = length_meter, 
                       key= elem['id'])     
    #node_id_to_coord_reverse = dict(zip(node_coord_to_id_dict.values(), 
    #                                 node_coord_to_id_dict.keys()))  
                           
    nx.set_node_attributes(G, 'node_id', node_coord_to_id_dict)
    return G


if __name__ == '__main__':
    eta_krr_net_shp = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS',
                    r'Py Codes\ETA_KRR\_files\LineString_Road_Network_UTM.shp')
    eta_G = CreateMultiGraph(eta_krr_net_shp)
    pickle.dump(eta_G, open('../_files/ETA_MultiGraph.p', 'wb'))
    