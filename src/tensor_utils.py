import numpy as np

def apply_2qubit(i,vec,V,random_phase=False,j=None,phase=0):
    ''' Apply 2-qubit tensor to indices (i,i+1) '''
    if j == None:
        j = i+1

    if phase or (random_phase and np.random.randint(2)):
        op = np.conj(V)
    else:
        op = V
        
    vec = np.tensordot(op,vec,axes=([2,3],[i,j]))
    vec = np.moveaxis(vec,[0,1],[i,j])
    return vec

def apply_1qubit(i,vec,U):
    ''' Apply 1-qubit tensor to index i '''
    vec = np.tensordot(U,vec,axes=(1,i))
    vec = np.moveaxis(vec,0,i)
    return  vec

def fan_out_leg(i,vec,nstates=2):
    """ Add leg next to index i """
    iden = np.zeros(nstates**3).reshape((nstates,nstates,nstates))
    for j in range(nstates):
        iden[j,j,j] = 1
    
    vec = np.tensordot(iden,vec,axes=(0,i))
    vec = np.moveaxis(vec,(0,1),(i,i+1))
    return  vec

def merge_legs(i,j,vec,nstates=2):
    """ Apply identity to indices i and j """
    iden = np.zeros(nstates**3).reshape((nstates,nstates,nstates))
    for k in range(nstates):
        iden[k,k,k] = 1
    
    vec = np.tensordot(iden,vec,axes=((0,1),(i,j)))
    vec = np.moveaxis(vec,0,i)
    return  vec

def trace_leg(i,vec):
    """ Trace out leg i """
    GHZ = np.ones(2)
    vec = np.tensordot(GHZ,vec,axes=([0],[i]))
    return vec

def add_EPR(i,vec,j=None):
    """ Add an EPR state to index i  """
    if j is None:
        j = i+1
        
    L0 = len(np.shape(vec))
        
    EPR = np.zeros((2,2),dtype=complex)
    EPR[0,0] = 1
    EPR[1,1] = 1

    vec = np.outer(EPR,vec).reshape([2]*(L0+2))
    vec = np.moveaxis(vec,(0,1),(i,j))
    
    return vec

def partial_trace(i,j,vec,nstates=2):
    iden = np.zeros(nstates**2).reshape((nstates,nstates))
    for k in range(nstates):
        iden[k,k] = 1
    
    vec = np.tensordot(iden,vec,axes=((0,1),(i,j)))
    return  vec

def add_leg(vec,i=0,vec2=None,nstates=2):
    """ Add a leg initialized in state vec2 """
    if vec2 is None:
        vec2 = [1]*nstates

    vec = np.kron(vec,vec2)    
    vec = vec.reshape([nstates]*(len(np.shape(vec))+1))

    vec = np.moveaxis(vec,len(np.shape(vec))-1,i)
    return  vec