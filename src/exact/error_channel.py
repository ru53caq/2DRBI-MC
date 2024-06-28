import numpy as np
import math
import scipy

from scipy.sparse import diags
from scipy.sparse.linalg import splu

# from kraus_operator import *
from kraus_operator import *
from tensor_utils import *
from surface_utils import *

class SingleSiteErrorChannel:
    """ A class representing an error channel on a single site. """
    def __init__(self, kraus_ops, probabilities):
        self.kraus_ops = []
        self.probabilities = []

        for (k,p) in zip(kraus_ops,probabilities):
            # skip negligible terms
            if p > 100*np.finfo(float).eps:
                self.kraus_ops.append(k)
                self.probabilities.append(p)

    def is_unital(self):
        return np.all([op.is_unitary() for op in self.kraus_ops])

    def is_unitary(self):
        return len(self.kraus_ops) == 1 and self.kraus_ops[0].is_unitary()

    def is_incoherent(self):
        return np.all([op.is_pauli() for op in self.kraus_ops])

    def is_uniaxial(self):
        return np.all([op.is_uniaxial() for op in self.kraus_ops])

    def get_pauli_twirl_approx(self):
        prob_PTA = np.zeros(4)

        for (p,op) in zip(self.probabilities, self.kraus_ops):
            prob_PTA += p*op.pauli_twirl_probabilities()

        return SingleSiteErrorChannel([pauli("I"),pauli("X"),pauli("Y"),pauli("Z")],prob_PTA)

    def __str__(self):
        label = ""
        for (op,p) in zip(self.kraus_ops,self.probabilities):
            if label != "":
                label += "\n"
            label += str(p) + ": " + str(op)

        return label

    def __repr__(self):
        return str([x for x in zip(self.probabilities,self.kraus_ops)])

class ErrorChannel:
    """ A class representing a full error channel factorized into single-site channels. """
    def __init__(self, d, error_channels):
        self.d = d
        self.error_channels = error_channels

        self.parity_matrix_x = get_parity_matrix(d)
        self.matching = get_matching(d)
        self.parity_matrix_z = self._transform_parity_matrix()
        self.ps = None
        self.thetas = None

    def _transform_parity_matrix(self):
        parity_matrix_z = np.array([np.fliplr(row.reshape(self.d,self.d).transpose()).flatten() 
                                        for row in self.parity_matrix], dtype=np.uint8)
        return parity_matrix_z

    def __repr__(self):
        return str(self.error_channels)

    def is_unital(self):
        return np.all([channel.is_unital() for channel in self.error_channels])

    def is_incoherent(self):
        return np.all([channel.is_incoherent() for channel in self.error_channels])

    def is_uniaxial(self):
        return np.all([channel.is_uniaxial() for channel in self.error_channels])

    def is_unitary(self):
        return np.all([channel.is_unitary() for channel in self.error_channels])

    def partition_function(self, bonds_x = None, bonds_z = None):
        if self.is_incoherent() and self.is_uniaxial():
            return self.partition_function_uniaxial(bonds_x)
        elif self.is_unitary() or self.is_incoherent():
            return self.partition_function_local(bonds_x,bonds_z)
        else:
            raise ValueError("Partition function for non-unital channel not currently supported.")

    def get_pauli_twirl_approx(self):
        return ErrorChannel(self.d, [n.get_pauli_twirl_approx() for n in self.error_channels])
    
    def partition_function_exact(self, bonds=None):
        '''Calculate partition function for small system size using
        brufe force enumeration of state space.

        Args:
            bonds: binary matrix specifying FM / AFM bonds

        Returns: 
            Logarithm of partition function under even / odd BCs
        '''
        def calc_energy(self):
            '''
            To calculate the energy, should take into a spin configuration
            In the whole project, we assume X error
            So syndromes are on the Z stabilizers

            '''
            z_bonds = self.parity_matrix_z.T @ self.z_stabs % 2  # suppose every bond is FM bond
            z_bonds ^= self.initial_error_config  # taking into account the AFM bonds
            return np.sum(z_bonds)

    def partition_function_uniaxial(self,bonds=None):
        ''' Calculates partition function for uniaxial error model 
        using exact tensor network contraction.

        Args:
            bonds: binary matrix specifying FM / AFM bonds

        Returns: 
            Logarithm of partition function under even / odd BCs
        '''
        d = self.d
        is_incoherent = self.is_incoherent()
        nstates = 2

        if bonds is None:
            bonds = [0]*(d**2)

        def bond_idx(i,j):
            return i*d+j
        
        L_nodes = d//2+1

        # tensor coefficients
        def get_coeffs(bond_idx):
            channel = self.error_channels[bond_idx]

            if is_incoherent:
                (c0,c1) = (0,0)
                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == identity():
                        c0 = coeff
                    elif op == sigmaz():
                        c1 = coeff
            else:
                (c0,c1) = (channel.kraus_ops[0].coeffs[0],channel.kraus_ops[0].coeffs[3])

            return (c0,c1)

        # set tensors from bond configuration
        def get_U(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                return np.array([[c0,c1],[c1,c0]])
            elif bond == 1:
                return np.array([[c1,c0],[c0,c1]])
            
        def get_V(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                return np.diag([c0,c1,c1,c0]).reshape((2,2,2,2))
            elif bond == 1:
                return np.diag([c1,c0,c0,c1]).reshape((2,2,2,2))

        # initialize MPS state
        vec = np.zeros(nstates**L_nodes,dtype=complex)
        vec[0] = 1
        vec = vec.reshape([nstates]*L_nodes)
        
        # perform contraction
        for depth in range(L_nodes-1):
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth)))
            vec = fan_out_leg(0,vec,nstates=nstates)
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth+1)))

            for i in range(1,L_nodes-1):
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i-1,2*depth)))
                vec = apply_1qubit(i,vec,get_U(bond_idx(2*i-1,2*depth+1)))
                vec = apply_1qubit(i+1,vec,get_U(bond_idx(2*i,2*depth)))
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i,2*depth+1)))

            vec = apply_2qubit(L_nodes-1,vec,get_V(bond_idx(d-2,2*depth)))
            vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-2,2*depth+1)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth+1)))
            vec = merge_legs(L_nodes-1,L_nodes,vec,nstates=nstates)
        
        for i in range(L_nodes-1):
            vec = apply_1qubit(i,vec,get_U(bond_idx(2*i,d-1)))
            vec = apply_2qubit(i,vec,get_V(bond_idx(2*i+1,d-1)))
        vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-1,d-1)))
        
        # project final state w.r.t. both boundary conditions
        if is_incoherent:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(vec[tuple([0]*L_nodes)]),np.log(vec[tuple([1]*L_nodes)]))
        else:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(np.abs(vec[tuple([0]*L_nodes)])**2),
                          np.log(np.abs(vec[tuple([1]*L_nodes)])**2))
        return errors

    def partition_function_uniaxial_temperature(self,bonds=None):
        #TODO: change the coefficients of local MPO to be temperature dependent
        ''' Calculates partition function for uniaxial error model 
        using exact tensor network contraction.

        Args:
            bonds: binary matrix specifying FM / AFM bonds

        Returns: 
            Logarithm of partition function under even / odd BCs
        '''
        d = self.d
        is_incoherent = self.is_incoherent()
        nstates = 2

        if bonds is None:
            bonds = [0]*(d**2)

        def bond_idx(i,j):
            return i*d+j
        
        L_nodes = d//2+1

        # tensor coefficients
        def get_coeffs(bond_idx):
            channel = self.error_channels[bond_idx]

            if is_incoherent:
                (c0,c1) = (0,0)
                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == identity():
                        c0 = coeff
                    elif op == sigmaz():
                        c1 = coeff
            else:
                (c0,c1) = (channel.kraus_ops[0].coeffs[0],channel.kraus_ops[0].coeffs[3])

            return (c0,c1)

        # set tensors from bond configuration
        def get_U(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                return np.array([[c0,c1],[c1,c0]])
            elif bond == 1:
                return np.array([[c1,c0],[c0,c1]])
            
        def get_V(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                return np.diag([c0,c1,c1,c0]).reshape((2,2,2,2))
            elif bond == 1:
                return np.diag([c1,c0,c0,c1]).reshape((2,2,2,2))

        # initialize MPS state
        vec = np.zeros(nstates**L_nodes,dtype=complex)
        vec[0] = 1
        vec = vec.reshape([nstates]*L_nodes)
        
        # perform contraction
        for depth in range(L_nodes-1):
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth)))
            vec = fan_out_leg(0,vec,nstates=nstates)
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth+1)))

            for i in range(1,L_nodes-1):
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i-1,2*depth)))
                vec = apply_1qubit(i,vec,get_U(bond_idx(2*i-1,2*depth+1)))
                vec = apply_1qubit(i+1,vec,get_U(bond_idx(2*i,2*depth)))
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i,2*depth+1)))

            vec = apply_2qubit(L_nodes-1,vec,get_V(bond_idx(d-2,2*depth)))
            vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-2,2*depth+1)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth+1)))
            vec = merge_legs(L_nodes-1,L_nodes,vec,nstates=nstates)
        
        for i in range(L_nodes-1):
            vec = apply_1qubit(i,vec,get_U(bond_idx(2*i,d-1)))
            vec = apply_2qubit(i,vec,get_V(bond_idx(2*i+1,d-1)))
        vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-1,d-1)))
        
        # project final state w.r.t. both boundary conditions
        if is_incoherent:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(vec[tuple([0]*L_nodes)]),np.log(vec[tuple([1]*L_nodes)]))
        else:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(np.abs(vec[tuple([0]*L_nodes)])**2),
                          np.log(np.abs(vec[tuple([1]*L_nodes)])**2))
        return errors

    def partition_function_uniaxial_mag(self, beta, h, bonds=None):
        #TODO: change the coefficients of local MPO to be magnetic field dependent
        ''' Calculates partition function for uniaxial error model 
        using exact tensor network contraction.

        Args:
            bonds: binary matrix specifying FM / AFM bonds

        Returns: 
            Logarithm of partition function under even / odd BCs
        '''
        d = self.d
        is_incoherent = self.is_incoherent()
        nstates = 2

        if bonds is None:
            bonds = [0]*(d**2)

        def bond_idx(i,j):
            return i*d+j
        
        L_nodes = d//2+1

        # tensor coefficients
        def get_coeffs(bond_idx):
            channel = self.error_channels[bond_idx]

            if is_incoherent:
                (c0,c1) = (0,0)
                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == identity():
                        c0 = coeff
                    elif op == sigmaz():
                        c1 = coeff
            else:
                (c0,c1) = (channel.kraus_ops[0].coeffs[0],channel.kraus_ops[0].coeffs[3])

            return (c0,c1)

        # set tensors from bond configuration
        def get_U(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                # return np.array([[c0,c1],[c1,c0]])
                return np.array([[c0,c1*np.exp(-2*beta*h)],[c1*np.exp(2*beta*h),c0]])
            elif bond == 1:
                return np.array([[c1,c0*np.exp(-2*beta*h)],[c0*np.exp(2*beta*h),c1]])
            
        def get_V(bond_idx):
            (c0,c1) = get_coeffs(bond_idx)
            bond = bonds[bond_idx]

            if bond == 0:
                return np.diag([c0,c1,c1,c0]).reshape((2,2,2,2))
            elif bond == 1:
                return np.diag([c1,c0,c0,c1]).reshape((2,2,2,2))

        # initialize MPS state
        vec = np.zeros(nstates**L_nodes,dtype=complex)
        vec[0] = 1
        vec = vec.reshape([nstates]*L_nodes)
        
        # perform contraction
        for depth in range(L_nodes-1):
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth)))
            vec = fan_out_leg(0,vec,nstates=nstates)
            vec = apply_1qubit(0,vec,get_U(bond_idx(0,2*depth+1)))

            for i in range(1,L_nodes-1):
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i-1,2*depth)))
                vec = apply_1qubit(i,vec,get_U(bond_idx(2*i-1,2*depth+1)))
                vec = apply_1qubit(i+1,vec,get_U(bond_idx(2*i,2*depth)))
                vec = apply_2qubit(i,vec,get_V(bond_idx(2*i,2*depth+1)))

            vec = apply_2qubit(L_nodes-1,vec,get_V(bond_idx(d-2,2*depth)))
            vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-2,2*depth+1)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth)))
            vec = apply_1qubit(L_nodes,vec,get_U(bond_idx(d-1,2*depth+1)))
            vec = merge_legs(L_nodes-1,L_nodes,vec,nstates=nstates)
        
        for i in range(L_nodes-1):
            vec = apply_1qubit(i,vec,get_U(bond_idx(2*i,d-1)))
            vec = apply_2qubit(i,vec,get_V(bond_idx(2*i+1,d-1)))
        vec = apply_1qubit(L_nodes-1,vec,get_U(bond_idx(d-1,d-1)))
        
        # project final state w.r.t. both boundary conditions
        if is_incoherent:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(vec[tuple([0]*L_nodes)]),np.log(vec[tuple([1]*L_nodes)]))
        else:
            with np.errstate(divide = 'ignore'):
                errors = (np.log(np.abs(vec[tuple([0]*L_nodes)])**2),
                          np.log(np.abs(vec[tuple([1]*L_nodes)])**2))
        return errors

    def partition_function_uniaxial_incoherent(self, bonds = None):
        ''' Calculates partition function using fermionic linear optics. 

       Based on: Bravyi et al, PRA (2014).
        ''' 
        d = self.d

        if bonds is None:
            bonds = np.array([0]*(d**2))
        
        # intialize the error probablity on each physical qubit
        ps = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                bond_idx = j*d+i
                channel = self.error_channels[bond_idx]

                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == sigmaz():
                        ps[i,j] = coeff

        def get_A(lambdas):
            A = np.zeros((len(lambdas)+1,len(lambdas)+1))
            for (i,l) in enumerate(lambdas):
                A[i+1,i] = -l
                A[i,i+1] = l
            return A

        def log_det(M):
            M = scipy.sparse.csc_matrix(M)
            lu = splu(M)
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
        
            return np.sum(np.log(np.abs(diagU)))+np.sum(np.log(np.abs(diagL)))

        def simulate_horizontal(j,bonds,M,log_gamma):
            ts = []
            ss = []
            for i in range(d):
                p = ps[i,j]

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

        def simulate_vertical(j,bonds,M,log_gamma):
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
            # B = np.diag(ss)
            B = scipy.sparse.diags(ss)
            # print(A)
            # print(B)
            
            log_gamma += log_det(M+A)/2
            M = A - B @ np.linalg.inv(M+A) @ B
            # M = A - B @ scipy.sparse.linalg.inv(M+A) @ B
            
            return (M,log_gamma)

        bonds_Z0 = bonds.copy()
        bonds_Z0 = bonds_Z0.reshape((d,d))
        bonds_Z0 = bonds_Z0.transpose()

        bonds_Z1 = bonds_Z0.copy()
        bonds_Z1[1,:] = (bonds_Z1[1,:] + 1) % 2

        bonds_vert = np.zeros((d-1,d-1))
        bonds_vert[::2,::2 ] = 1
        bonds_vert[1::2,1::2 ] = 1
        bonds_vert = bonds_vert.transpose()
        
        for Z in [0,1]:
            if Z == 0:
                bonds = bonds_Z0
            else:
                bonds = bonds_Z1

            log_pi0 = 0
            for i in range(d):
                for j in range(d):
                    p = ps[i,j]
                    log_pi0 += np.log((1-p))*(1-bonds[i,j])
                    log_pi0 += np.log((p))*(bonds[i,j])

            # log_pi0 = np.log((1-p))*(d**2-(np.sum(bonds)))
            # log_pi0 += np.log(p)*(np.sum(bonds))

            M0 = np.zeros((2*d,2*d))
            # M0 = scipy.sparse.lil_matrix((2*d,2*d))
            M0[0,-1] = 1
            M0[-1,0] = -1
            for i in range(1,2*d-2,2):
                M0[i,i+1] = 1
                M0[i+1,i] = -1
            
            # M0 = M0.tocsc()

            M = M0.copy()
            log_gamma = np.log(2)*(d-1)

            for j in range(d-1):
                (M,log_gamma) = simulate_horizontal(j,bonds,M,log_gamma)
                # print(np.exp(log_gamma))
                # print()
                (M,log_gamma) = simulate_vertical(j,bonds_vert,M,log_gamma)

            (M,log_gamma) = simulate_horizontal(d-1,bonds,M,log_gamma)
            # print(np.linalg.det(M))
            # print(np.exp(log_gamma))
            
            log_Z = log_pi0
            log_Z += log_gamma/2 - np.log(2)/2
            log_Z += log_det(M+M0)/4

            if Z == 0:
                log_Z0 = log_Z
            else:
                log_Z1 = log_Z

        return (log_Z0,log_Z1)

    def partition_function_uniaxial_incoherent_temperature(self, beta, bonds = None):
        ''' Calculates partition function using fermionic linear optics. 

       Based on: Bravyi et al, PRA (2014).
        ''' 
        d = self.d

        if bonds is None:
            bonds = np.array([0]*(d**2))
        
        # intialize the error probablity on each physical qubit
        ps = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                bond_idx = j*d+i            
                channel = self.error_channels[bond_idx]

                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == sigmaz():
                        ps[i,j] = coeff

        def get_A(lambdas):
            A = np.zeros((len(lambdas)+1,len(lambdas)+1))
            for (i,l) in enumerate(lambdas):
                A[i+1,i] = -l
                A[i,i+1] = l
            return A

        def log_det(M):
            M = scipy.sparse.csc_matrix(M)
            lu = splu(M)
            diagL = lu.L.diagonal()
            diagU = lu.U.diagonal()
        
            return np.sum(np.log(np.abs(diagU)))+np.sum(np.log(np.abs(diagL)))

        def simulate_horizontal(j,bonds,M,log_gamma):
            ts = []
            ss = []
            for i in range(d):
                p = ps[i,j]

                if bonds[i,j] == 0:  # FM bond
                    w = np.exp(-2*beta)
                    # w = np.exp(-beta)
                else:  # AFM bond
                    w = np.exp(2*beta)
                    # w = np.exp(beta)

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

        def simulate_vertical(j,bonds,M,log_gamma):
            ts = [0]
            ss = [1]
            for i in range(d-1):
                if bonds[i,j] == 0:
                    w = 1
                    # w = np.exp(2*beta)
                else:
                    w = 0
                    # w = np.exp(-2*beta)

                log_gamma += np.log((1+w**2))
                
                t = 2*w/(1+w**2)
                s = (1-w**2)/(1+w**2)

                ts.append(t)
                ts.append(0)
                
                ss.append(s)
                ss.append(s)
            
            ss.append(1)

            A = get_A(ts)
            # B = np.diag(ss)
            B = scipy.sparse.diags(ss)
            # print(A)
            # print(B)
            
            log_gamma += log_det(M+A)/2
            M = A - B @ np.linalg.inv(M+A) @ B
            # M = A - B @ scipy.sparse.linalg.inv(M+A) @ B
            
            return (M,log_gamma)

        bonds_Z0 = bonds.copy()
        bonds_Z0 = bonds_Z0.reshape((d,d))
        bonds_Z0 = bonds_Z0.transpose()

        bonds_Z1 = bonds_Z0.copy()
        bonds_Z1[1,:] = (bonds_Z1[1,:] + 1) % 2

        bonds_vert = np.zeros((d-1,d-1))
        bonds_vert[::2,::2 ] = 1
        bonds_vert[1::2,1::2 ] = 1
        bonds_vert = bonds_vert.transpose()
        
        for Z in [0,1]:
            if Z == 0:
                bonds = bonds_Z0
            else:
                bonds = bonds_Z1

            log_pi0 = 0
            for i in range(d):
                for j in range(d):
                    p = ps[i,j]
                    log_pi0 += np.log((1-p))*(1-bonds[i,j])
                    log_pi0 += np.log((p))*(bonds[i,j])

            # log_pi0 = np.log((1-p))*(d**2-(np.sum(bonds)))
            # log_pi0 += np.log(p)*(np.sum(bonds))

            M0 = np.zeros((2*d,2*d))
            # M0 = scipy.sparse.lil_matrix((2*d,2*d))
            M0[0,-1] = 1
            M0[-1,0] = -1
            for i in range(1,2*d-2,2):
                M0[i,i+1] = 1
                M0[i+1,i] = -1
            
            # M0 = M0.tocsc()

            M = M0.copy()
            log_gamma = np.log(2)*(d-1)

            for j in range(d-1):
                (M,log_gamma) = simulate_horizontal(j,bonds,M,log_gamma)
                # print(np.exp(log_gamma))
                # print()
                (M,log_gamma) = simulate_vertical(j,bonds_vert,M,log_gamma)

            (M,log_gamma) = simulate_horizontal(d-1,bonds,M,log_gamma)
            # print(np.linalg.det(M))
            # print(np.exp(log_gamma))
            
            log_Z = log_pi0
            log_Z += log_gamma/2 - np.log(2)/2
            log_Z += log_det(M+M0)/4

            if Z == 0:
                log_Z0 = log_Z
            else:
                log_Z1 = log_Z

        return (log_Z0,log_Z1)

    def partition_function_local(self, bonds_x=None, bonds_z=None):
        ''' Calculates partition function for local (non-uniaxial) error model 
        using exact tensor network contraction.

        Args:
            bonds: binary matrix specifying FM / AFM bonds

        Returns: 
            Logarithm of partition function under even / odd BCs in both 
            horizontal and vertical directions
        '''
        d = self.d
        is_incoherent = self.is_incoherent()
        nstates = 2

        if bonds_x is None:
            bonds_x = [0]*(d**2)

        if bonds_z is None:
            bonds_z = [0]*(d**2)

        def bond_idx(i,j):
            return i*d+j

        # tensor coefficients
        def get_coeffs(bond_idx):
            channel = self.error_channels[bond_idx]

            if is_incoherent:
                (c0,c1,c2,c3) = (0,0,0,0)
                for (op,p) in zip(channel.kraus_ops,channel.probabilities):
                    if op == identity():
                        c0 = p
                    elif op == sigmax():
                        c1 = p
                    elif op == sigmay():
                        c2 = p
                    elif op == sigmaz():
                        c3 = p
            else:
                (c0,c1,c2,c3) = (channel.kraus_ops[0].coeffs[0],
                                 channel.kraus_ops[0].coeffs[1],
                                 1j*channel.kraus_ops[0].coeffs[2],
                                 channel.kraus_ops[0].coeffs[3])

            return (c0,c1,c2,c3)

        # set tensors from bond configuration
        def get_U(bond_idx):
            (c0,c1,c2,c3) = get_coeffs(bond_idx)
            bond_x = bonds_x[bond_idx]
            bond_z = bonds_z[bond_idx]

            if bond_x == 0 and bond_z == 1:
                (c0,c1,c2,c3) = (c3,c2,c1,c0)
            elif bond_x == 1 and bond_z == 0:
                (c0,c1,c2,c3) = (c1,c0,c3,c2)
            elif bond_x == 1 and bond_z == 1:
                (c0,c1,c2,c3) = (c2,c3,c0,c1)

            U =   np.array([[c0,c3,c1,c2],
                            [c1,c2,c0,c3],
                            [c3,c0,c2,c1],
                            [c2,c1,c3,c0]]).reshape((2,2,2,2))

            return U

        def get_V(bond_idx):
            (c0,c1,c2,c3) = get_coeffs(bond_idx)
            bond_x = bonds_x[bond_idx]
            bond_z = bonds_z[bond_idx]

            if bond_x == 0 and bond_z == 1:
                (c0,c1,c2,c3) = (c3,c2,c1,c0)
            elif bond_x == 1 and bond_z == 0:
                (c0,c1,c2,c3) = (c1,c0,c3,c2)
            elif bond_x == 1 and bond_z == 1:
                (c0,c1,c2,c3) = (c2,c3,c0,c1)

            V =   np.array([[c0,c1,c3,c2],
                            [c3,c2,c0,c1],
                            [c1,c0,c2,c3],
                            [c2,c3,c1,c0]]).reshape((2,2,2,2))

            return V

        # projections
        P0 =  np.diag([1,0]).reshape((2,2))
        P1 =  np.diag([0,1]).reshape((2,2))

        # loop over even / odd boundary conditions in vertical direction
        for vertical_BC in [0,1]:
            vec = np.zeros(2**(d//2+2),dtype=complex)
            vec[0] = 1
            vec = vec.reshape([2]*(d//2+2))

            if vertical_BC == 0:
                P = P0
            else:
                P = P1
                
            for depth in range(d):
                # project top leg onto 0 state
                if depth%2 == 0:
                    vec = apply_1qubit(0,vec,P0)

                # apply layer of tensors
                for i in range(d):
                    if i < d-1 and ((depth > 0) or i%2 == 0):
                        if i == 0:
                            fan_i = 1
                        else:
                            fan_i = 2+i

                        vec = fan_out_leg(fan_i,vec)
                    elif depth == 0 and i%2 == 1:
                        vec = add_EPR(2+i,vec)

                    if i == 0:
                        op_i = 0
                    else:
                        op_i = i+1

                    if (i+depth)%2 == 0:
                        op = get_U(bond_idx(depth,i))
                    else:
                        op = get_V(bond_idx(depth,i))

                    vec = apply_2qubit(op_i,vec,op)

                    if i > 0:
                        vec = merge_legs(i,1+i,vec)

                # project bottom leg
                if depth < d-1 and depth%2 == 0:
                    vec = apply_1qubit(d,vec,P)
                    
                if depth == d-1:
                    for i in range(d//2):
                        vec = trace_leg(i+1,vec)

            # project final state based on horizontal boundary condition
            if vertical_BC == 0:
                Z00 = vec[tuple([0]*(d//2+1)+[vertical_BC])]
                Z01 = vec[tuple([1]*(d//2+1)+[vertical_BC])]
            else:
                Z10 = vec[tuple([0]*(d//2+1)+[vertical_BC])]
                Z11 = vec[tuple([1]*(d//2+1)+[vertical_BC])]

        if is_incoherent:
            with np.errstate(divide = 'ignore'):
                errors = np.log(np.abs(np.array([Z00,Z01,Z10,Z11])))
        else:
            with np.errstate(divide = 'ignore'):
                errors = np.log(np.abs(np.array([Z00,Z01,Z10,Z11]))**2)

        return errors

    def sample_config_incoherent(self):
        """ Sample an error configuration from incoherent distribution. """
        if self.ps is None:
            self.ps = [] 
            for channel in self.error_channels:
                for (op,coeff) in zip(channel.kraus_ops,channel.probabilities):
                    if op == sigmaz():
                        self.ps.append(coeff)

        return [np.random.binomial(1, p) for p in self.ps]

    def monte_carlo_incoherent_with_measurement(self, num_iters, p_measure, repetitions):
        """ Perform Monte Carlo simulation with incoherent errors.

        Args:
            num_iters: number of Monte Carlo iterations
            p_measure: measurement error rate
            repetitions: number of measurements repetitions

        Returns:
            Error under MPWM decoder
        """
        d = self.d
        H = self.parity_matrix

        matching = get_matching(d,repetitions=repetitions)

        num_stabilisers, num_qubits = H.shape

        num_errors = 0
        for _ in range(num_iters):
            noise_new = []
            for _ in range(repetitions):
                noise_new += [self.sample_config_incoherent()]
            noise_new = np.array(noise_new).transpose()

            noise_cumulative = (np.cumsum(noise_new, 1) % 2).astype(np.uint8)
            noise_total = noise_cumulative[:,-1]

            syndrome = H@noise_cumulative % 2
            syndrome_error = (np.random.rand(num_stabilisers, repetitions) < p_measure).astype(np.uint8)
            syndrome_error[:,-1] = 0 # Perfect measurements in last round to ensure even parity

            noisy_syndrome = (syndrome + syndrome_error) % 2
            # Convert to difference syndrome
            noisy_syndrome[:,1:] = (noisy_syndrome[:,1:] - noisy_syndrome[:,0:-1]) % 2

            predicted_logicals_flipped = matching.decode(noisy_syndrome)
            error = (noise_total ^ predicted_logicals_flipped)
            if np.sum(error[:d]) % 2 == 1:
                num_errors += 1

        return num_errors/num_iters

    def partition_function_uniaxial_mps(self, bonds, chi, scale=1):
        # stabilizer = np.random.randint(2, size=(2*nqubit, nqubit-1))
        d = self.d
        As = []  # a
        ps_flip = []   # p
        thetas = []
        for channel in self.error_channels:
            p_incoh = 0 
            p_flip = 0 

            for (op,p) in zip(channel.kraus_ops,channel.probabilities):
                if op.is_pauli():
                    p_incoh += p
                    if op == sigmaz():
                        p_flip = p
                else:
                    theta = op.get_theta()
                    thetas.append(theta)

            As.append(1-p_incoh)

            if p_incoh > 0:
                ps_flip.append(p_flip/p_incoh)
            else:
                ps_flip.append(0)

        p = np.mean(ps_flip)
        if p == 0:
            p = np.mean(np.sin(theta)**2)
        a = np.mean(As)

        # print((p,a))

        nqubit = self.d//2+1
        Z0, Z1 = parfunc(bonds.reshape((d,d)).transpose(), nqubit, p, a*np.sqrt(p/(1-p)), scale=scale, chi=chi)
        # print((Z0,Z1))
        return (np.log(np.abs(Z0)),np.log(np.abs(Z1)))

    def run_ideal_decoder_uniaxial(self, syndrome, min_steps, max_steps):
        d = self.d

        As = []  # a
        ps_flip = []   # p
        for channel in self.error_channels:
            p_incoh = 0 
            p_flip = 0 

            for (op,p) in zip(channel.kraus_ops,channel.probabilities):
                if op.is_pauli():
                    p_incoh += p
                    if op == sigmaz():
                        p_flip = p
                else:
                    theta = op.get_theta()

            As.append(1-p_incoh)

            if p_incoh > 0:
                ps_flip.append(p_flip/p_incoh)
            else:
                ps_flip.append(0)

        ps_flip = np.array(ps_flip)
        As = np.array(As)

        ps_incoh = 1/2-1/2*np.sqrt(1-4*(1-As**2)*(1-ps_flip)*ps_flip)
        thetas = np.arcsin(np.sqrt(1/2-1/2*(1-2*ps_flip)/np.sqrt(1-4*(1-As**2)*(1-ps_flip)*ps_flip)))

        channel_incoherent = ErrorChannel(d,[single_site_dephasing_channel(p) for p in ps_incoh])
        channel_coherent = ErrorChannel(d,[single_site_rotation_channel(theta) for theta in thetas])

        s = 0
        n = 0
        log_Z = -np.inf

        correction = self.matching.decode(syndrome) # minimum weight perfect match
        incoherent_bonds = np.array([np.random.binomial(1, p) for p in ps_incoh])

        for i in range(max_steps):
            new_incoherent_bonds = np.array([np.random.binomial(1, p) for p in ps_incoh])
            
            correction2 = correction ^ new_incoherent_bonds
            
            # log_Z0,log_Z1 = partition_function_uncorrelated_c4(theta,L,correction2.reshape(L**2),thetas=thetas)
            try:
                log_Z0_new,log_Z1_new = channel_coherent.partition_function_c4(correction2)
                
                log_Z_new = np.log(1+np.exp(log_Z1_new-log_Z0_new))+log_Z0_new
                
                if (1-np.exp(log_Z-log_Z_new) > 100*np.finfo(float).eps) or np.random.rand() < np.exp(log_Z_new-log_Z):
                    # print(np.exp(log_Z))
                    log_Z = log_Z_new
                    log_Z0 = log_Z0_new
                    log_Z1 = log_Z1_new

                    incoherent_bonds = new_incoherent_bonds

                    log_Z0_incoh,log_Z1_incoh = channel_incoherent.partition_function_uniaxial_incoherent(incoherent_bonds)
                    log_Z_incoh = np.log(1+np.exp(log_Z1_incoh-log_Z0_incoh))+log_Z0_incoh

                    P0_incoh = np.exp(log_Z0_incoh-log_Z_incoh)
                    P1_incoh = np.exp(log_Z1_incoh-log_Z_incoh)
                    # print(P0_incoh,P1_incoh)
                 
                if i > min_steps:
                    # error = np.exp(log_Z1-log_Z)
                    error = np.exp(log_Z1-log_Z)*P0_incoh
                    error += np.exp(log_Z0-log_Z)*P1_incoh

                    s += error
                    n += 1
            except:
                n += 0

        avg = s/n
        return ((avg < .5),avg)

    def run_ideal_decoder_uniaxial2(self, syndrome, min_steps, max_steps):
        d = self.d

        As = []  # a
        ps_flip = []   # p
        for channel in self.error_channels:
            p_incoh = 0 
            p_flip = 0 

            for (op,p) in zip(channel.kraus_ops,channel.probabilities):
                if op.is_pauli():
                    p_incoh += p
                    if op == sigmaz():
                        p_flip = p
                else:
                    theta = op.get_theta()

            As.append(1-p_incoh)

            if p_incoh > 0:
                ps_flip.append(p_flip/p_incoh)
            else:
                ps_flip.append(0)

        ps_flip = np.array(ps_flip)
        As = np.array(As)

        ps_incoh = 1/2-1/2*np.sqrt(1-4*(1-As**2)*(1-ps_flip)*ps_flip)
        thetas = np.arcsin(np.sqrt(1/2-1/2*(1-2*ps_flip)/np.sqrt(1-4*(1-As**2)*(1-ps_flip)*ps_flip)))

        channel_incoherent = ErrorChannel(d,[single_site_dephasing_channel(p) for p in ps_incoh])
        channel_coherent = ErrorChannel(d,[single_site_rotation_channel(theta) for theta in thetas])

        s = 0
        n = 0
        log_Z = -np.inf

        correction = self.matching.decode(syndrome) # minimum weight perfect match
        incoherent_syndrome = self.parity_matrix @ np.array([np.random.binomial(1, p) for p in ps_incoh]) % 2
        # incoherent_bonds = np.array([np.random.binomial(1, p) for p in ps_incoh])
        # incoherent_bonds = self.matching.decode(incoherent_syndromes)

        for i in range(max_steps):
            new_incoherent_syndrome = incoherent_syndrome.copy()
            new_incoherent_syndrome[np.random.randint(len(incoherent_syndrome))] ^= 1
            new_incoherent_bonds = self.matching.decode(new_incoherent_syndrome)

            log_Z0_incoh, log_Z1_incoh = channel_incoherent.partition_function_uniaxial_incoherent(new_incoherent_bonds)
            # new_incoherent_bonds = np.array([np.random.binomial(1, p) for p in ps_incoh])
            
            correction2 = correction ^ new_incoherent_bonds
            
            # log_Z0,log_Z1 = partition_function_uncorrelated_c4(theta,L,correction2.reshape(L**2),thetas=thetas)
            try:
                log_Z0_coh,log_Z1_coh = channel_coherent.partition_function_c4(correction2)

                log_Z0_new = np.log(np.exp(log_Z0_coh+log_Z0_incoh)+np.exp(log_Z1_coh+log_Z1_incoh))
                log_Z1_new = np.log(np.exp(log_Z1_coh+log_Z0_incoh)+np.exp(log_Z0_coh+log_Z1_incoh))
                
                log_Z_new = np.log(1+np.exp(log_Z1_new-log_Z0_new))+log_Z0_new
                
                if (1-np.exp(log_Z-log_Z_new) > 100*np.finfo(float).eps) or np.random.rand() < np.exp(log_Z_new-log_Z):
                    # print(np.exp(log_Z))
                    log_Z = log_Z_new
                    log_Z0 = log_Z0_new
                    log_Z1 = log_Z1_new

                    incoherent_syndrome = new_incoherent_syndrome

                    # log_Z0_incoh,log_Z1_incoh = channel_incoherent.partition_function_uniaxial_incoherent(incoherent_bonds)
                    # log_Z_incoh = np.log(1+np.exp(log_Z1_incoh-log_Z0_incoh))+log_Z0_incoh

                    # P0_incoh = np.exp(log_Z0_incoh-log_Z_incoh)
                    # P1_incoh = np.exp(log_Z1_incoh-log_Z_incoh)
                 
                if i > min_steps:
                    # error = np.exp(log_Z1-log_Z)
                    # error = np.exp(log_Z1-log_Z)*P0_incoh
                    # error += np.exp(log_Z0-log_Z)*P1_incoh
                    error = np.exp(log_Z1-log_Z)

                    s += error
                    n += 1
            except:
                n += 1

        avg = s/n
        return ((avg < .5),avg)

    def monte_carlo_incoherent(self, num_iters, incoherent_decoder = False):
        d = self.d

        total_error_min_weight = 0
        total_error_incoherent = 0
        n = 0

        for _ in range(num_iters):
            # incoherent_bonds = np.random.binomial(1, args.p*(1-args.a**2)/(1-args.p*args.a**2), L**2)
            incoherent_bonds = np.array(self.sample_config_incoherent())

            syndrome = self.parity_matrix @ incoherent_bonds % 2
            correction = self.matching.decode(syndrome) # minimum weight perfect match

            error = correction ^ incoherent_bonds
            error = error.reshape((d,d))

            try:
                if np.sum(error[0,:])%2 > 0:
                    total_error_min_weight += 1

                if incoherent_decoder:
                    log_Z0, log_Z1 = self.partition_function_uniaxial_incoherent(incoherent_bonds)
                    log_Z = np.log(1+np.exp(log_Z1-log_Z0))+log_Z0 
                    n += 1

                    total_error_incoherent += np.minimum(np.exp(log_Z1-log_Z),np.exp(log_Z0-log_Z))
            except:
                n += 0

        total_error_min_weight /= num_iters

        if incoherent_decoder:
            total_error_incoherent /= n


        return (total_error_min_weight, total_error_incoherent)

    def metropolis_uniaxial(self,num_sweeps, num_eq=1000, num_flips=10):
        """ Perform metropolis sampling over syndromes.

        Args:    
            num_sweeps: number of steps for the measurement
            num_eq:     number of equilibration steps before the start of the measurement 
            num_flip:   number of steps between measurements

        Returns:
            Logical error rate for ideal decoder and minimum weight decoder
        """
        d = self.d

        W_syn = d//2+1
        L_syn = d-1

        error = 0
        total_error_ideal = 0
        total_error_min_weight = 0

        log_Z = np.nan
        syndrome = np.random.binomial(1, 0.5, self.parity_matrix.shape[0])

        diffs = []
        syndromes = []

        for step in range(num_sweeps+num_eq):
            for _ in range(num_flips):
                new_syndrome = syndrome.copy()

                new_syndrome[np.random.randint(len(syndrome))] ^= 1

                correction = self.matching.decode(new_syndrome) # minimum weight perfect match

                # compute partition function
                log_Z0_new,log_Z1_new = self.partition_function_uniaxial(correction)
                log_Z_new = np.log(1+np.exp(log_Z1_new-log_Z0_new))+log_Z0_new 

                ratio = np.exp(log_Z-log_Z_new)
                
                # Metropolis update
                if np.isnan(log_Z) or ((1-ratio) > 100*np.finfo(float).eps) or (np.random.rand() < 1/ratio):
                    log_Z = log_Z_new
                    log_Z0 = log_Z0_new
                    log_Z1 = log_Z1_new

                    syndrome = new_syndrome
                    
            if step > num_eq:
                # for ideal decoder, choose smaller partition function
                error_ideal = np.minimum(np.exp(log_Z1-log_Z),np.exp(log_Z0-log_Z))

                # for minimum weight decoder, always choose symmetric b.c. 
                error_min_weight = np.exp(log_Z1-log_Z)
                
                total_error_ideal += error_ideal/num_sweeps
                total_error_min_weight += error_min_weight/num_sweeps

        return (total_error_ideal, total_error_min_weight)

    def metropolis_local(self, num_sweeps, num_flips=10, num_eq=1000):
        """ Perform metropolis sampling over syndromes.

        Args:    
            num_sweeps: number of steps for the measurement
            num_eq:     number of equilibration steps before the start of the measurement 
            num_flip:   number of steps between measurements

        Returns:
            Logical error rate for ideal decoder, minimum weight decoder, and ideal classical decoder 
        """
        d = self.d

        W_syn = d//2+1
        L_syn = d-1

        channel_PTA = self.get_pauli_twirl_approx()

        error = 0
        total_error_ideal = 0
        total_error_min_weight = 0
        total_error_classical = 0

        Z = 0
        syndrome_x = np.zeros(W_syn*L_syn,dtype=np.uint8)
        syndrome_z = np.zeros(W_syn*L_syn,dtype=np.uint8)

        diffs = []
        syndromes = []

        for step in range(num_sweeps+num_eq):
            for _ in range(num_flips):
                new_syndrome_x = syndrome_x.copy()
                new_syndrome_z = syndrome_z.copy()

                if np.random.rand() < 0.5:
                    new_syndrome_x[np.random.randint(len(syndrome_x))] ^= 1
                if np.random.rand() < 0.5:
                    new_syndrome_x[np.random.randint(len(syndrome_x))] ^= 1

                if np.random.rand() < 0.5:
                    new_syndrome_z[np.random.randint(len(syndrome_z))] ^= 1
                if np.random.rand() < 0.5:
                    new_syndrome_z[np.random.randint(len(syndrome_z))] ^= 1

                bonds_x = self.matching.decode(new_syndrome_x) # minimum weight perfect match
                bonds_z = self.matching.decode(new_syndrome_z).reshape((d,d)) # minimum weight perfect match
                bonds_z = bonds_z[:,::-1].transpose().flatten()


                log_Zs_new = self.partition_function_local(bonds_x,bonds_z)
                log_Zs_classical_new = channel_PTA.partition_function_local(bonds_x,bonds_z)

                Zs_new = np.exp(log_Zs_new)
                Zs_classical_new = np.exp(log_Zs_classical_new)

                Z_new = np.sum(np.abs(Zs_new))
                
                if Z_new > 0 and ((1-np.abs(Z/Z_new) > 100*np.finfo(float).eps) or np.random.rand() < Z_new/Z):
                    Z = Z_new
                    Zs = Zs_new
                    Zs_classical = Zs_classical_new

                    syndrome_x = new_syndrome_x
                    syndrome_z = new_syndrome_z
                    
            if step > num_eq:
                # for ideal decoder, choose maximum partition function
                error_ideal = (Z-np.max(np.abs(Zs)))/Z

                # for minimum weight decoder, always choose symmetric b.c. 
                error_min_weight = (Z-np.abs(Zs[0]))/Z

                # decode with optimal incoherent decoder
                error_classical = (Z-np.abs(Zs[np.argmax(Zs_classical)]))/Z
                
                total_error_ideal += error_ideal/num_sweeps
                total_error_min_weight += error_min_weight/num_sweeps
                total_error_classical += error_classical/num_sweeps

        return (total_error_ideal,total_error_min_weight,total_error_classical)


def single_site_dephasing_channel(p):
    return SingleSiteErrorChannel([identity(), sigmaz()],[(1-p),p])

def dephasing_channel(d,p):
    return ErrorChannel(d,[single_site_dephasing_channel(p) for _ in range(d**2)])

def single_site_rotation_channel(theta, alpha=0, beta=0):
    return SingleSiteErrorChannel([KrausOp(theta=theta,alpha=alpha,beta=beta)],[1])

def uniform_rotation_channel(d,theta,alpha=0,beta=0):
    return ErrorChannel(d,[single_site_rotation_channel(theta,alpha,beta) for _ in range(d**2)])
