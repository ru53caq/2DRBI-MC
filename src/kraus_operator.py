import numpy as np
import math

class KrausOp:
    """ A class representing a single-site Kraus operator """

    def __init__(self, coeffs=None, theta=None, alpha=0, beta=0):
        """Initialize via either Pauli coefficients or spherical angles.

        coeffs: coefficients of the 4 Pauli operators (I,X,Y,Z)

        theta: angle of rotation
        alpha: polar angle
        beta: azimuthal angle
        """
        if coeffs is not None:
            if len(coeffs) != 4:
                raise ValueError('KrausOp must have 4 coefficients for (I,X,Y,Z).')

            self.coeffs = np.array(coeffs)
        else:
            self.coeffs = np.array([np.cos(theta),
                                    1j*np.sin(theta)*np.sin(alpha)*np.cos(beta),
                                    1j*np.sin(theta)*np.sin(alpha)*np.sin(beta),
                                    1j*np.sin(theta)*np.cos(alpha)])

    def is_unitary(self):
        if np.abs(self.coeffs[0]) > 0:
            is_unitary = np.isreal(self.coeffs[0])
            is_unitary &= np.all(np.isreal(1j*self.coeffs[1:]))
        else:
            is_unitary = np.all(np.isreal(self.coeffs[1:]/np.sum(self.coeffs[1:])))

        is_unitary &= math.isclose(np.abs(np.linalg.norm(self.coeffs)),1)

        return is_unitary

    def is_uniaxial(self):
        """ Whether channel is uniaxial w.r.t. Z basis """
        return math.isclose(np.linalg.norm(self.coeffs[1:3]),0)

    def is_pauli(self):
        return math.isclose(np.abs(np.sum(self.coeffs)),1) and math.isclose(np.abs(np.sum(self.coeffs**2)),1)

    def get_theta(self):
        if not self.is_unitary():
            raise ValueError("theta is not defined for a non-unitary channel") 

        return np.arcsin(np.imag(self.coeffs[3]))

    def pauli_twirl_probabilities(self):
        """ Returns probabilities of (I,X,Y,Z) of Pauli-twirled channel """
        return np.abs(self.coeffs)**2

    def __repr__(self):
        label = ""
        for (coeff,pauli) in zip(self.coeffs,["I","X","Y","Z"]):
            if coeff != 0:
                if label != "":
                    label += " + "
                label += str(coeff) + "*" + pauli

        return label

    def __eq__(self, other):
        # remove overall phase
        coeffs1 = self.coeffs/np.sum(self.coeffs)*np.abs(np.sum(self.coeffs))
        coeffs2 = other.coeffs/np.sum(other.coeffs)*np.abs(np.sum(other.coeffs))

        return np.all(coeffs1 == coeffs2)

def identity():
    return KrausOp([1,0,0,0])

def sigmax():
    return KrausOp([0,1,0,0])

def sigmay():
    return KrausOp([0,0,1,0])

def sigmaz():
    return KrausOp([0,0,0,1])

def pauli(op):
    if op == "I":
        return identity()
    elif op == "X":
        return sigmax()
    elif op == "Y":
        return sigmay()
    elif op == "Z":
        return sigmaz()
    else:
        raise ValueError('Invalid Pauli operator')


