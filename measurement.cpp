#include "measurement.hpp"

// determine the op to measure; it may also be realized by combining vector of std::function with std::map.
double ising_measurement::measure(std::string op, std::vector<int> s) {
    if(op == "magFM")
        return this->magFM(s);
    else
        { std::cout << "Unknown order parameter! \n" << std::endl;
        throw std::runtime_error("Unknown order parameter!"); } //err msg maybe not shown with osx
}


/** Magnetizations **/
double ising_measurement::magFM(std::vector<int> s) {

    double mag = 0.;

    for (int i = 0; i < N_sites; i+=1)
    	mag += s[i];

    return mag / N_sites;
}