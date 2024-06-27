import numpy as np
import pymatching as pm
import networkx as nx
import matplotlib.pyplot as plt

from surface_utils import *
from error_channel import *

class MarkovChain(ErrorChannel):
    def __init__(self, d, error_channels, beta):
        super().__init__(d, error_channels)
        self.beta = beta  # inverse temperature
        # transform the parity check matrix for X stabilizers to that of Z stabilizers
        self.parity_matrix_z = self._transform_parity_matrix()
        self.initial_error_config = np.zeros(self.parity_matrix.shape[1], dtype=np.uint8)
        # configuration of X stabilizers should be determined by the initial error configuration
        self.x_stabs = np.zeros(self.parity_matrix.shape[0], dtype=np.uint8)
        self.z_stabs = np.zeros(self.parity_matrix.shape[0], dtype=np.uint8)
        self.energy = 0.
        
    def _transform_parity_matrix(self):
        parity_matrix_z = np.array([np.fliplr(row.reshape(d,d).transpose()).flatten() 
                                        for row in self.parity_matrix], dtype=np.uint8)
        return parity_matrix_z

    def init_chain(self, error_config):
        '''Initialize the error configuration and spin model (energy function) of the Markov chain.'''
        self.initial_error_config = error_config
        self.x_stabs = self.parity_matrix @ self.initial_error_config  # this one should stay the same across the whole process
        self.z_stabs = np.zeros(self.parity_matrix.shape[0], dtype=np.uint8)
        self.energy = self.calc_energy()
        
    def calc_energy(self):
        z_bonds = self.parity_matrix_z.T @ self.z_stabs % 2  # suppose every bond is FM bond
        z_bonds ^= self.initial_error_config  # taking into account the AFM bonds
        return np.sum(z_bonds)
    
    def propose(self):
        """Propose a random application of Z stabilizers (spin-flip)
        """        
        return np.random.randint(0, len(self.z_stabs))

    def update(self):
        pos = self.propose()
        delta_z_bonds = self.parity_matrix_z[pos]
        delta_E = np.sum(delta_z_bonds)
        # the case delta_E < 0 is already taken into account for the expression below
        if np.random.rand() < np.exp(-self.beta * delta_E):
            self.z_stabs[pos] ^= 1
            self.energy += delta_E

def partition_function_mc(chain, burn_in=10, num_steps=10, num_samples=1000):
    """
    Compute the partition function using Markov chain Monte Carlo.
    """
    Z = 0.
    for _ in range(burn_in):
        for _ in range(num_steps):
            chain.update()
        chain.update()
    for _ in range(num_samples):
        for _ in range(num_steps):
            chain.update()
        Z += np.exp(-chain.beta * chain.energy)
    return Z / num_samples
    
def mc_decoding_logical_error_rate(chain, num_error_configs=1000):
    '''Sample multiple error configurations and decode them using Markov chain Monte Carlo.'''
    num_decoding_error = 0
    for _ in range(num_error_configs):
        error_config_actual = chain.sample_config_incoherent()
        error_config0 = error_config_actual.copy() # error configuration after MWPM correction
        
        # partition function for error configuration 0
        chain.init_chain(error_config0)
        Z0 = chain.partition_function_mc()
        
        # the vector that switches the boundary condition
        boundary_switch = np.zeros((d,d), dtype=np.uint8)
        boundary_switch[:,1] = 1
        boundary_switch.reshape((d**2,))
        
        # partition function for error configuration 1
        error_config1 = error_config_actual.copy() ^ boundary_switch
        chain.init_chain(error_config1)
        Z1 = chain.partition_function_mc()
        
        # decide which error configuration is more likely
        if Z1 > Z0:
            num_decoding_error += 1
    return num_decoding_error / num_error_configs

def update_all_chains(chains, localsteps=10, parallel=False):
    if parallel:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count())
        def update_chain(chain):
            for _ in range(localsteps):
                chain.update()
        pool.map(update_chain, chains)   
    else:
        for chain in chains:
            for _ in range(localsteps):
                chain.update()

def swap(chains):
    for i in range(0, len(chains)-1):
        p_switch = np.exp((chains[i].beta-chains[i+1].beta)*(chains[i].energy-chains[i+1].energy))
        # p_switch = min(1, p_switch) # no need to do that in practice
        if np.random.rand() < p_switch:
            chains[i].z_stabs = chains[i+1].z_stabs.copy()
            chains[i+1].z_stabs = chains[i].z_stabs.copy()
            chains[i].energy, chains[i+1].energy = chains[i+1].energy, chains[i].energy

def partition_function_parallel_tempering(d, error_channels, beta, num_chains=4, burn_in=10, num_steps=10, num_samples=1000):
    Z = 0.
    betas = np.linspace(0, beta, num_samples)
    chains = [MarkovChain(d, error_channels, beta) for beta in betas]
    for chain in chains:
        chain.init_chain(chain.sample_config_incoherent())
    for _ in range(burn_in):
        update_all_chains(chains, localsteps=num_steps, parallel=False)
    for _ in range(num_samples):
        update_all_chains(chains, localsteps=num_steps, parallel=False)
        swap(chains)
    # only cares about the chain of the lowest temperature
    Z += np.exp(-chains[-1].beta * chains[-1].energy)
    return Z / num_samples

# TODO: use tolerance window to determine the stop of the algorithm

p = 0.1
d = 5
beta = 0.5 * np.log((1-p)/p)  # Nishimori condition
error_channels = [single_site_dephasing_channel(p) for _ in range(d**2)]
num_steps = 300
chain = MarkovChain(d, error_channels, beta)
error_config = chain.sample_config_incoherent()
chain.init_chain(error_config)


##################################################
# DEBUG PRINTING
##################################################