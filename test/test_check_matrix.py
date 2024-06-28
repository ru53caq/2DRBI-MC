import sys
import os
import unittest
import numpy as np
from numpy.typing import NDArray
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'exact')
))
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src', 'exact')
))
from surface_utils import get_x_check_matrix, get_z_check_matrix
from error_channel import *

class TestSurfaceUtils(unittest.TestCase):
    """Test get_x_check_matrix and get_z_check_matrix for distance 5."""
    def test_dist5(self):
        x_check_matrix: NDArray[np.int_] = get_x_check_matrix(5)
        z_check_matrix = get_z_check_matrix(5)

        # Test X check matrix size
        self.assertEqual(len(x_check_matrix), 12)
        self.assertEqual(len(x_check_matrix[0]), 25)

        # Test the X check matrix connectivity
        # rows correspond to X checks,
        # each row should have 2 or 4 non-zero entries
        for row in range(len(x_check_matrix)): 
            self.assertIn(np.sum(x_check_matrix[row]), [2, 4])
        # columns correspond to qubits,
        # each column should have 1 or 2 non-zero entries
        for col in range(len(x_check_matrix[0])):
            self.assertIn(np.sum(x_check_matrix[:, col]), [1, 2])

        # Finer test of the Z check matrix connectivity
        # In general, the expected qubits should be 
        d = 5
        expected_single_connection_qubit_indices = set(
            list(range(0, d**2-d+1, d)) + list(range(d-1, d**2, d))
        )
        self.assertSetEqual(set(
            np.nonzero(np.sum(z_check_matrix, axis=0) == 1)[0]
            ),expected_single_connection_qubit_indices
        )

        # Test the 0th X check
        expected_qubit_indices = set([0, 5])
        self.assertSetEqual(set(
            np.nonzero(x_check_matrix[0])[0]
            ), expected_qubit_indices
        )

        # Test the 7th X check
        expected_qubit_indices = set([11, 12, 16, 17])
        self.assertSetEqual(set(
            np.nonzero(x_check_matrix[7])[0]
            ), expected_qubit_indices
        )

        # Test the 0th Z check
        expected_qubit_indices = set([0, 1, 5, 6])
        self.assertSetEqual(set(
            np.nonzero(z_check_matrix[0])[0]
            ), expected_qubit_indices
        )

        # Test the 10th Z check
        z_check_matrix = get_z_check_matrix(5)
        expected_qubit_indices = set([8, 9, 13, 14])
        self.assertSetEqual(set(
            np.nonzero(z_check_matrix[10])[0]
            ), expected_qubit_indices
        )

    def test_dist_d(self):
        d = 9

        x_check_matrix: NDArray[np.int_] = get_x_check_matrix(d)
        z_check_matrix = get_z_check_matrix(d)

        # Test X check matrix size
        self.assertEqual(len(x_check_matrix), (d-1)*(d//2+1))
        self.assertEqual(len(x_check_matrix[0]), d**2)

        # Test the X check matrix connectivity
        expected_single_connection_qubit_indices = set(
            list(range(0, d**2-d+1, d)) + list(range(d-1, d**2, d))
        )
        self.assertSetEqual(set(
            np.nonzero(np.sum(z_check_matrix, axis=0) == 1)[0]
            ), expected_single_connection_qubit_indices
        )
        expected_double_connection_qubit_indices = set(range(0, d**2)) \
                                            - set(list(range(0, d**2-d+1, d)) \
                                            + list(range(d-1, d**2, d)))
        self.assertSetEqual(set(
            np.nonzero(np.sum(z_check_matrix, axis=0) == 2)[0]
            ), expected_double_connection_qubit_indices
        )

if __name__ == '__main__':
    unittest.main()

