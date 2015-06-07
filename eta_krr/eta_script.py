'''
Created on May 5, 2015

@author: asifrehan
'''
import os
import networkx as nx
from networkx.utils import dict_to_numpy_array
import pickle
import  numpy as np
#==============================================================================
#1. done - clean up the shpfile to focus only on transit route
#2. construct new nx_graph
#    - replace trans_graph node_keys with graph keys
#    - edges > 999+
#    - nodes < -999
#3. make crowd sourcing agent
#4. construct Q_arr, ys 
#    - edge_key = graph[node1][node2].keys()
#5. Solve for regularizer and f
#6. Predict
#==============================================================================
def get_line_graph(prim_gr):
    return nx.line_graph(prim_gr)

def mk_dist_matrix(trans_graph):
    d = nx.shortest_path_length(trans_graph)
    idseq = [int(key[2]) for key in d.keys()]
    mapping = dict(zip(d.keys(), idseq))
    dist_mat = dict_to_numpy_array(d, mapping=mapping)
    return dist_mat

def aff_def_func(dist, cutoff=3, weight=0.5):
    if dist <= cutoff:
        return weight**dist
    else:
        return 0

def aff_mat(dist_mat):
    vec_calc_afnty = np.vectorize(aff_def_func)
    return vec_calc_afnty(dist_mat)

def get_degree_mat(affinity_matrix):
    return np.diag(np.sum(affinity_matrix, axis=1))

def get_lapl(dist_mat):
    aff = aff_mat(dist_mat)
    deg = get_degree_mat(aff)
    return deg - aff

def main(nx_graph):
    dist_matrix = mk_dist_matrix(get_line_graph(nx_graph))
    Lapl = get_lapl(dist_matrix)
    return Lapl

if __name__ == '__main__':
    eta_mg = os.path.join(r'C:\Users\asr13006\Google Drive\UConn MS\Py Codes',
                          r'HMM_Krumm_Newson_Implementation\MM_AR',
                          r'Relevant_files\MultiGraph.p')
    Lapl = main( pickle.load(open(eta_mg, 'rb')) )
    pickle.dump(Lapl, open(r'Laplacian_matrix.p', 'wb'))
    