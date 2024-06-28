import numpy as np
import pymatching
import networkx as nx
from typing import List, Tuple, Literal
from numpy.typing import NDArray

def get_parity_matrix(d,periodic=False):
    """ Returns parity matrix of tilted surface code.
    It's the X check matrix, namely it only checks Z errors.
    """
    if d == 3 and periodic:
        return np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 1, 1, 0, 0, 0],
                         [1, 1, 0, 1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 0, 0]])

    g = get_matching_graph(d,periodic=periodic)
    
    W_nodes = d//2+1
    L_nodes = d-1
    num_nodes = L_nodes*W_nodes
    
    parity_matrix = np.zeros((num_nodes,d**2), dtype=np.uint8)
    
    # add edges from graph
    for n in g.nodes():
        # boundary node
        if n == num_nodes:
            continue
            
        for e in g.edges(n):
            qubit = g[e[0]][e[1]]['fault_ids']
            parity_matrix[n,qubit] = 1
            
    # additional boundary edges
    if not periodic:
        for i in range(1,d,2):
            parity_matrix[i//2+1,i] = 1
            parity_matrix[i//2+W_nodes*(L_nodes-1),i+d*(d-1)] = 1
        
    return parity_matrix

def get_matching_graph(d,periodic=False):
    ''' Returns matching graph for local errors. 

    rounds: number of rounds of syndrome measurments
    
    Returns:
        Graph whose nodes are the stabilizer measurements and edges are errors
    '''
    W_nodes = d//2+1
    L_nodes = d-1
    num_nodes = L_nodes*W_nodes

    if not periodic:
        num_qubits = d**2
    else:
        num_qubits = d*(d-1)
    
    def node_idx(i,j,k=0):
        # k = measurement round
        return k*num_nodes+i*W_nodes+j
    
    def qubit_idx(i,j,k=0):
        # k = measurement round
        if not periodic:
            return k*num_qubits+i*d+j
        else:
            # print((i*L+j)%(L**2-L))
            return k*num_qubits+(i*d+j)%(d**2-d)
    
    boundary_node = num_nodes
    
    g = nx.Graph()
    
    # for k in range(rounds):
    k = 0
    if not periodic:
        # top row
        for j in range(0,d,2): 
            g.add_edge(node_idx(0,j//2,k), boundary_node, fault_ids=qubit_idx(0,j,k), weight=1)

        # bottom row
        for j in range(0,d,2): 
            g.add_edge(node_idx(L_nodes-1,j//2,k), boundary_node, fault_ids=qubit_idx(d-1,j,k), weight=1)

        g.nodes[boundary_node]['is_boundary'] = True
    else:
        for j in range(0,d,2): 
            g.add_edge(node_idx(0,j//2,k), node_idx(L_nodes-1,j//2,k), fault_ids=qubit_idx(0,j,k), weight=1)

        for j in range(1,d,2): 
            g.add_edge(node_idx(0,j//2+1,k), node_idx(L_nodes-1,j//2,k), fault_ids=qubit_idx(0,j,k), weight=1)
    
    # interior rows
    for i in range(1,d-1,2):
        for j in range(0,d,2): 
            g.add_edge(node_idx(i-1,j//2,k),node_idx(i,j//2,k), fault_ids=qubit_idx(i,j,k), weight=1)
        for j in range(1,d,2): 
            g.add_edge(node_idx(i-1,j//2+1,k),node_idx(i,j//2,k), fault_ids=qubit_idx(i,j,k), weight=1)
    for i in range(2,d-1,2):
        for j in range(0,d,2): 
            g.add_edge(node_idx(i-1,j//2,k),node_idx(i,j//2,k), fault_ids=qubit_idx(i,j,k), weight=1)
        for j in range(1,d,2): 
            g.add_edge(node_idx(i-1,j//2,k),node_idx(i,j//2+1,k), fault_ids=qubit_idx(i,j,k), weight=1)

        # measurement errors
        # if measurement_errors:
        #     if k > 0:
        #         for i in range(L_nodes):
        #             for j in range(W_nodes):
        #                 g.add_edge(node_idx(i,j,k-1),node_idx(i,j,k), 
        #                     fault_ids=num_qubits*rounds+k*num_nodes+node_idx(i,j), weight=1)
       
    return g

def get_matching(d, repetitions=None):
    if repetitions is None:
        return pymatching.Matching(get_matching_graph(d))
    else:
        return pymatching.Matching(get_parity_matrix(d), weights=1,
            repetitions=repetitions, timelike_weights=1)

def get_x_check_matrix(d, periodic=False) -> NDArray[np.int_]:
    """ Returns X parity check matrix of tilted surface code.
    Don't fit PyMatching format.
    """
    W_nodes = d//2+1
    L_nodes = d-1
    num_nodes = W_nodes*L_nodes
    pc = np.zeros((num_nodes,d**2), dtype=np.uint8)

    def _x_check_idx(i,j):
        return i*W_nodes+j

    def _qubit_idx(i,j):
        if not periodic:
            return i*d+j
        else:
            return (i*d+j)%(d**2-d)
    
    if periodic==True:
        raise NotImplementedError('Periodic boundary conditions not implemented yet')
    else:
        for i in range(d-1):
            if i%2 == 0:
                pc[_x_check_idx(i,0), _qubit_idx(i,0)] = 1  # upper qubit
                pc[_x_check_idx(i,0), _qubit_idx(i+1,0)] = 1  # lower qubit
                for j in range(1, d-1, 2):
                    pc[_x_check_idx(i, (j+1)//2), _qubit_idx(i,j)] = 1  # upper left qubit
                    pc[_x_check_idx(i, (j+1)//2), _qubit_idx(i,j+1)] = 1  # upper right qubit
                    pc[_x_check_idx(i, (j+1)//2), _qubit_idx(i+1,j)] = 1  # lower left qubit
                    pc[_x_check_idx(i, (j+1)//2), _qubit_idx(i+1,j+1)] = 1  # lower right qubit
            else:
                for j in range(0, d-1, 2):
                    pc[_x_check_idx(i, j//2), _qubit_idx(i,j)] = 1  # upper left qubit
                    pc[_x_check_idx(i, j//2), _qubit_idx(i,j+1)] = 1  # upper right qubit
                    pc[_x_check_idx(i, j//2), _qubit_idx(i+1,j)] = 1  # lower left qubit
                    pc[_x_check_idx(i, j//2), _qubit_idx(i+1,j+1)] = 1  # lower right qubit
                pc[_x_check_idx(i,(d-1)//2), _qubit_idx(i,d-1)] = 1  # upper qubit
                pc[_x_check_idx(i,(d-1)//2), _qubit_idx(i+1,d-1)] = 1  # lower qubit
    return pc

def get_z_check_matrix(d, periodic=False) -> NDArray[np.int_]:
    """ Returns Z parity check matrix of tilted surface code.
    Don't fit PyMatching format.
    """
    if periodic==True:
        raise NotImplementedError('Periodic boundary conditions not implemented yet')

    W_nodes = d//2+1
    L_nodes = d-1
    num_nodes = W_nodes*L_nodes
    pc = np.zeros((num_nodes,d**2), dtype=np.uint8)

    def _z_check_idx(i,j):
        return i + j*W_nodes

    def _qubit_idx(i,j):
        if not periodic:
            return i*d+j
        else:
            return (i*d+j)%(d**2-d)
    
    for j in range(d-1):
        if j%2 == 0:
            for i in range(0, d-1, 2):
                pc[_z_check_idx(i//2, j), _qubit_idx(i,j)] = 1
                pc[_z_check_idx(i//2, j), _qubit_idx(i,j+1)] = 1
                pc[_z_check_idx(i//2, j), _qubit_idx(i+1,j)] = 1
                pc[_z_check_idx(i//2, j), _qubit_idx(i+1,j+1)] = 1
            pc[_z_check_idx((d-1)//2, j), _qubit_idx(d-1,j)] = 1
            pc[_z_check_idx((d-1)//2, j), _qubit_idx(d-1,j+1)] = 1
        else:
            pc[_z_check_idx(0, j), _qubit_idx(0, j)] = 1
            pc[_z_check_idx(0, j), _qubit_idx(0, j+1)] = 1
            for i in range(1, d-1, 2):
                pc[_z_check_idx((i+1)//2, j), _qubit_idx(i,j)] = 1
                pc[_z_check_idx((i+1)//2, j), _qubit_idx(i,j+1)] = 1
                pc[_z_check_idx((i+1)//2, j), _qubit_idx(i+1,j)] = 1
                pc[_z_check_idx((i+1)//2, j), _qubit_idx(i+1,j+1)] = 1
    return pc


# ------------------------------------- Classical decoder ---------------------------------------------

def get_A(lambdas):
    A = np.zeros((len(lambdas)+1,len(lambdas)+1))
    for (i,l) in enumerate(lambdas):
        A[i+1,i] = -l
        A[i,i+1] = l
    return A

def log_det(M):
    lu = splu(M)
    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()
    
    return np.sum(np.log(np.abs(diagU)))+np.sum(np.log(np.abs(diagL)))

def simulate_horizontal(j,p,d,bonds,M,log_gamma):
    ts = []
    ss = []
    for i in range(d):
        if bonds[i,j] == 0:
            w = p/(1-p)
        else:
            w = (1-p)/p

        log_gamma += np.log((1+w**2)/2)
        
        t = (1-w**2)/(1+w**2)
        s = 2*w/(1+w**2)

        ts.append(t)
        ss.append(s)
        ss.append(s)

        if i < d-1:
            ts.append(0)

    A = get_A(ts)
    B = scipy.sparse.diags(ss)
    
    log_gamma += log_det(M+A)/2
    M = A - B @ np.linalg.inv(M+A) @ B
    
    return (M,log_gamma)

def simulate_vertical(j,p,d,bonds,M,log_gamma):
    ts = [0]
    ss = [1]
    for i in range(d-1):
        if bonds[i,j] == 0:
            w = 1
        else:
            w = 0

        log_gamma += np.log((1+w**2))
        
        t = 2*w/(1+w**2)
        s = (1-w**2)/(1+w**2)

        ts.append(t)
        ts.append(0)
        
        ss.append(s)
        ss.append(s)
    
    ss.append(1)

    A = get_A(ts)
    B = scipy.sparse.diags(ss)
    
    log_gamma += log_det(M+A)/2
    M = A - B @ np.linalg.inv(M+A) @ B
    
    return (M,log_gamma)

