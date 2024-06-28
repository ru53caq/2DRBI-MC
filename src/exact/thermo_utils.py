import numpy as np
from numpy.typing import NDArray
from error_channel import *


def gen_AFM_bonds(d, p, seed) -> NDArray[np.int_]:
    """0 for FM bond, 1 for AFM bond"""
    num_bonds = d**2
    rng = np.random.default_rng(seed)
    bonds = rng.binomial(1, p, num_bonds)  # 0 for FM bond, 1 for AFM bond
    bonds = bonds.astype(int)
    # convert to -1&1 representation
    bonds = 1 - 2*bonds  # 1 for FM bond, -1 for AFM bond
    # print(bonds)
    return bonds

def gen_AFM_boonds_special(d) -> NDArray[np.int_]:
    """
    Generate special bond configuration,
    where the two boundaries are AFM bonds and the rest are FM bonds.
    """
    num_bonds = d**2
    bonds = np.ones(num_bonds, dtype=int)
    bonds[list(range(0, d**2-d+1, d))] = -1
    bonds[list(range(d-1, d**2, d))] = -1
    return bonds

# jit can't handle integer arrays
def energy(
        z_check_matrix: NDArray[np.int_],
        spins: NDArray[np.int_],
        AFM_bonds: NDArray[np.int_],
        d: int,
        bc: Literal['even', 'odd']) -> float:
    """Energy function for the random bond Ising model (RBIM)
    Spins are the Z checks. 
    Indices of spins correspond to the row indices of z_check_matrix.
    AFM bonds are qubit errors.
    Indices of bonds correspond to the column indices of z_check_matrix.
    """
    energy = 0.
    # bulk
    bulk_mask = list(
        set(range(0, d**2)) - set(list(range(0, d**2 - d + 1, d))
        + list(range(d - 1, d**2, d)))
    )
    # In axis 1: 0 for spin index 1, 1 for spin index 2, 2 for bond type
    interaction_pairs = np.zeros((len(bulk_mask), 3), dtype=int)  
    
    for icol, col in enumerate(bulk_mask):
        interaction_pairs[icol][:-1] = np.where(z_check_matrix[:, col])[0]
        interaction_pairs[icol][-1] = AFM_bonds[col]  # keep bond type
    
    for pair in interaction_pairs:
        # if the bond type is FM (+1), 
        # the energy is reduced when the spins are aligned; 
        # if the bond type is AFM (-1),
        # the energy is reduced when the spins are anti-aligned
        energy -= spins[pair[0]] * spins[pair[1]] * pair[-1]
    
    # left boundary
    left_boundary_mask = list(range(0, d**2-d+1, d))  # left boundary
    # In axis 1: 0 for spin index, 1 for bond type
    left_boundary_pairs = np.zeros((d, 2), dtype=int)
    for icol, col in enumerate(left_boundary_mask):
        left_boundary_pairs[icol][0] = np.where(z_check_matrix[:, col])[0]
        left_boundary_pairs[icol][1] = AFM_bonds[col]

    for pair in left_boundary_pairs:
        # assume the left spin is +1
        # if the bond type is FM (+1),
        # the energy is reduced when the spins are aligned;
        energy -= spins[pair[0]]*pair[-1]

    # right boundary
    right_boundary_mask = list(range(d-1, d**2, d))  # right boundary
    # In axis 1: 0 for spin index, 1 for bond type
    right_boundary_pairs = np.zeros((d, 2), dtype=int)
    for icol, col in enumerate(right_boundary_mask):
        right_boundary_pairs[icol][0] = np.where(z_check_matrix[:, col])[0]
        # if bc is even, the left boundary treatment is the same as the bulk;
        # otherwise, it's the opposite
        right_boundary_pairs[icol][1] = AFM_bonds[col] if bc == 'even' \
                                        else (-1)*AFM_bonds[col]

    for pair in right_boundary_pairs:
        # if the bond type is FM (+1),
        # the energy is reduced when the spins are aligned;
        energy -= spins[pair[0]]*pair[-1]
    
    return energy

def calc_thermodyn(z_check_matrix: NDArray[np.int_],
                   AFM_bonds: NDArray[np.int_],
                   beta: float,
                   d: int,
                   bc: Literal['even', 'odd']):
    """Partition function.
    Energy expectation value.
    Magnetization expectation value.
    Enumerate all possible spin configurations and sum over Boltzmann factors.
    """
    print(beta)
    pf = 0.
    energy_expct_numerator = 0.
    mag_expct_numerator = 0.
    # spins is a binary vector of length (d-1)*(d//2+1),
    # enumerating all possible spin configurations
    num_spins = (d-1)*(d//2+1)
    for i in range(2**num_spins):
        # spins with 0&1 representation
        spins = np.array([int(x) for x in np.binary_repr(i, width=num_spins)])  
        # convert spins to -1&1 representation 
        spins = 1 - 2*spins
        enrg = energy(z_check_matrix, spins, AFM_bonds, d, bc)
        boltzmann_factor = np.exp(-beta * enrg)
        pf += boltzmann_factor
        energy_expct_numerator += enrg * boltzmann_factor
        mag = np.sum(spins)
        mag_expct_numerator += mag * boltzmann_factor
    return pf, energy_expct_numerator/pf, mag_expct_numerator/pf
