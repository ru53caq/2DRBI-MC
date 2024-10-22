/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_TUTORIALS_MC_ISING2_ISING_HPP_2e5dd7c49f7f4f209a3d27964d231a5b
#define ALPS_TUTORIALS_MC_ISING2_ISING_HPP_2e5dd7c49f7f4f209a3d27964d231a5b

#include <alps/accumulators.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/mc/mcbase.hpp>

#include "storage_type.hpp"

#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/multi_array.hpp>
#include <boost/function.hpp>

#include <Eigen/Dense>

#include "pt_adapter.hpp"
#include "temperature.hpp"

#include "measurement.hpp"  
#include "square_rotated.hpp"                                              


// Simulation class for 2D Ising model (square lattice).
// Extends alps::mcbase, the base class of all Monte Carlo simulations.
// Defines its state, calculation functions (update/measure) and
// serialization functions (save/load)


class ising_sim : public pt_adapter<phase_space_point::temperature> {
    using Base = pt_adapter<phase_space_point::temperature>;
    // The internal state of our simulation
    


  private:

    int N_replica;
    int L;
    int lat_sites;
    lattice::square lat;// define lattice    
    int pt_sweeps;
    std::vector<int> S;
    phase_point temp;

    int Nmeas = 0;
    double E_tot;
    int NBins;
    int PTval;
    std::string simdir;
    std::vector<int> J_x;
    std::vector<int> J_y;
    std::vector<double> T_vec;
    std::vector<double> p_vec;
    double T;
    double p;
    int ind;
    bool up=true;
    int N_core;
    std::vector<double> time_in_Ti;

    // Convergence check
    bool converged = false;

    //PT acceptance rate
    double pt_checker=0;
    std::vector<int> this_J_x;
    std::vector<int> this_J_y;
    std::vector<int> next_J_x;
    std::vector<int> next_J_y;
    double this_T;
    double this_p;
    double next_T;
    double next_p;
    ising_measurement op_measure; // order parameters and constraint
    std::string orders; // orders to measure

    /* Sweep and control */
    int sweeps; // current sweep after thermailization
    int total_sweeps; // maximum number of sweeps
    int thermalization_sweeps;
    int measuring_sweeps;

    std::vector<std::string> op_names;
    std::vector<std::string> parsing_op(std::string str);

    std::string initial_state;

    double sampling_range_a; // defined to for flexibility
    double sampling_range_b;

    std::mt19937 rng;
    std::mt19937 rng2;
    std::uniform_int_distribution<int> random_site;
    std::uniform_int_distribution<int> rand_L;

  public:

    ising_sim(parameters_type & parms, std::size_t seed_offset = 0);

    static void define_parameters(parameters_type & parameters);

    double total_energy();
    double mag_FM();
    double local_energy(int i);

    void initialization(std::string str);
    void Heatbath(int i, double beta);
    void overrelaxation();
    
    void record_measurement();

    virtual void update() ;
    virtual void measure();
    virtual double fraction_completed() const;

    using alps::mcbase::save;
    using alps::mcbase::load;
    virtual void save(alps::hdf5::archive & ar) const;
    virtual void load(alps::hdf5::archive & ar);


    virtual void reset_sweeps(bool skip_therm) override;
    virtual phase_point phase_space_point() const override;
    virtual bool update_phase_point(phase_point const&) override;

    std::vector<double> configuration() const;
    std::vector<double> random_configuration();

};

#endif /* ALPS_TUTORIALS_MC_ISING2_ISING_HPP_2e5dd7c49f7f4f209a3d27964d231a5b */
