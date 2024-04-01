/* Define measurement of order parameters */
#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>


#include <Eigen/Dense>
#include "square_rotated.hpp"

constexpr double pi(boost::math::constants::pi<double>());


class ising_measurement{
public:
    ising_measurement() {}
    ising_measurement(size_t L): L(L), N_sites(L*L), lat(L) {}

private:

    int L;
    int N_sites;
    lattice::square lat;// lattice; needs to be constructed by L

public:
    double measure(std::string op, std::vector<int> s); // determine the op to measure
    /* Magnetizations; verifed */
    double magFM(std::vector<int> s);
};
