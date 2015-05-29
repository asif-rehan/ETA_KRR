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
#4. construct Q, ys 
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
    #==========================================================================
    # for i in range(len(d)):
    #     for j in range(len(d)):
    #         if i != j and dist_mat[i][j] == 0 :
    #             dist_mat[i][j] = np.inf
    #==========================================================================
    return dist_mat

#==============================================================================
#aff_def_func mobilizes function for affinity
#vectorizes the function to apply on distance matrix of line graph
#==============================================================================
def aff_def_func(dist, cutoff=3, weight=0.5):
    if dist <= cutoff:
        return weight**dist
    else:
        return 0
#==============================================================================
# here calculates and returns L
#==============================================================================
def aff_mat(dist_mat):
    vec_calc_afnty = np.vectorize(aff_def_func)
    return vec_calc_afnty(dist_mat)

def get_degree_mat(affinity_matrix):
    return np.diag(np.sum(affinity_matrix, axis=1))

def get_lapl(dist_mat):
    aff = aff_mat(dist_mat)
    deg = get_degree_mat(aff)
    return deg - aff

# [int(x[2]) for x in nx.shortest_path_length(nx.line_graph(mg)).keys()]
#==============================================================================
# creates crowd sourcing agent  
#==============================================================================
def cs_agent():
    
    return path_x, trvl_tm_y 

def makeQ():
    return Q
#==============================================================================
# KRR for trajectory regression
#==============================================================================
def LOO_cost(Q, M, Ns, lam, L, yn_vec):
    Im = np.identity(M)
    H = np.dot(np.dot(n),Q)
    temp = np.dot(np.invert(np.diag((Im - H))), np.dot((Im - H), yn_vec)) 
    LOO_cost = 
    return reg_paramM

def solve_RLS(reg_param):
    return vec_f

if __name__ == '__main__':
    eta_mg = os.path.join('/media/asifrehan/shared_data_folder/Google Drive',
                          'UConn MS/Py Codes/ETA_KRR/_files/ETA_MultiGraph.p')
    mg = pickle.load(open(eta_mg, 'rb'))
    dist_matrix = mk_dist_matrix(get_line_graph(mg))
    Lapl = get_lapl(dist_matrix)