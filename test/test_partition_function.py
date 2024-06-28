import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'exact')
))
from numpy.testing import assert_allclose
import thermo_utils
import surface_utils

class TestPartitionFunction(unittest.TestCase):
    def test_pf_ratios_zero_afm(self):
        """
        Notes: bond config file for MC:
        L_5/p_0.000/even/Seed_0/config_p=0.000.data
        Warning: This test only passes when tol is set to 1e-2.
        1e-3 is too small.
        Means we should improve precision in the future.
        """
        d = 5
        bc = 'even'
        z_check_matrix = surface_utils.get_z_check_matrix(d)
        num_bonds = d**2
        betas = 1/np.array([4., 4.5, 5., 5.5, 6.])
        mc_ratios = [1.27842, 1.18214, 1.13614, 1.09732]
        bond_config = np.ones(num_bonds, dtype=int)
        exact_pfs = []
        for beta in betas:
            pf, _, _ =  thermo_utils.calc_thermodyn(z_check_matrix, 
                                                    bond_config, beta, d, bc)
            exact_pfs.append(pf)
        exact_ratios = [exact_pfs[i]/exact_pfs[i+1] 
                        for i in range(len(exact_pfs)-1)]
        assert_allclose(exact_ratios, mc_ratios, rtol=1e-2)

    def test_pf_ratios_special_bond_config(self):
        d = 5
        bc = 'even'
        z_check_matrix = surface_utils.get_z_check_matrix(d)
        num_bonds = d**2
        betas = 1/np.array([4., 4.5, 5., 5.5, 6.])
        mc_ratios = [1.24375, 1.17069, 1.12404, 1.09183]
        bond_config = np.ones(num_bonds, dtype=int)
        bond_config[3] = -1
        bond_config[8] = -1
        exact_pfs = []
        for beta in betas:
            pf, _, _ =  thermo_utils.calc_thermodyn(z_check_matrix, 
                                                    bond_config, beta, d, bc)
            exact_pfs.append(pf)
        exact_ratios = [exact_pfs[i]/exact_pfs[i+1] 
                        for i in range(len(exact_pfs)-1)]
        assert_allclose(exact_ratios, mc_ratios, rtol=1e-2)

