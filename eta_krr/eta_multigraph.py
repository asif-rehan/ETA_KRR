import networkx as nx
import fiona, pickle
import os
#import matplotlib.pyplot as plt
from fiona.crs import from_epsg

def CreateMultiGraph(utm_shp_path):
    G = nx.MultiGraph()
    with fiona.open(utm_shp_path,crs= from_epsg(32618)) as shp:
        #driver='ESRI Shapefile'
        for elem in shp:
            strt = elem['geometry']['coordinates'][0]
            len_coords = len(elem['geometry']['coordinates'])
            end = elem['geometry']['coordinates'][len_coords-1]
            length_meter = elem['properties']['Length_met']
            G.add_edge(strt, end, 
                       weight = length_meter, 
                       key= elem['id'])
    return G

def CreateMultiDiGraph(utm_shp_path):
    G = nx.MultiDiGraph()
    with fiona.open(utm_shp_path,crs= from_epsg(32618)) as shp:
        #driver='ESRI Shapefile'
        for elem in shp:
            strt = elem['geometry']['coordinates'][0]
            len_coords = len(elem['geometry']['coordinates'])
            end = elem['geometry']['coordinates'][len_coords-1]
            length_meter = elem['properties']['Length_met']
            G.add_edge(strt, end, weight = length_meter)
            G.add_edge(end, strt, weight = length_meter)                                    
    return G


if __name__ == '__main__':
    utm_shp_path = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS',
                    r'Py Codes\ETA_KRR\_files\LineString_Road_Network_UTM.shp')
    G = CreateMultiGraph(utm_shp_path)
    pickle.dump(G, open('../_files/ETA_MultiGraph.p', 'wb'))
    eta_MG = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS',
                    r'Py Codes\ETA_KRR\_files\ETA_MultiGraph.p')
    open(eta_MG)