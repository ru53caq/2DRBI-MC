/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "2DRBI.hpp"
#include "mpi.hpp"
#include "temperature.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <alps/hdf5.hpp>
#include <alps/params.hpp>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

int main(int argc, char** argv)
{
    // Define the type for the simulation
    typedef ising_sim my_sim_type;
    using temperature = phase_space_point::temperature;
    
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    mpi::communicator comm_world;

    const bool is_master = (comm_world.rank() == 0);

    try {
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        if (is_master)
            std::cout << "Creating simulations..." << std::endl;
        
        alps::params parameters(argc, argv);
        my_sim_type::define_parameters(parameters);
        
        
        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }
        
        int N_replica = (int)(parameters["N_replica"]);

        //start with a different Seed so we have a different physical realization of the system in every new simulation
        parameters["SEED"]=int(parameters["SEED"]);

        my_sim_type sim(parameters,1*comm_world.rank());
        // sim.rebind_communicator(comm_world);
        
        // If needed, restore the last checkpoint
        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();
        if(comm_world.rank() != 0) checkpoint_file+="."+std::to_string(comm_world.rank());
        if (parameters.is_restored()) {
            std::cout << "Restoring checkpoint from " << checkpoint_file << std::endl;
            sim.load(checkpoint_file);
        }
        int L = parameters["L"];
        double dis = parameters["disorder"];
        std::string init_state = parameters["initial_state"];
        std::ifstream Tp_points("../L_" + std::to_string(L) + "/" + "p_"+std::to_string(dis).substr(0,5) + "/"  + init_state + "/Seed_0/T-p_points.data");

        std::vector<double> T_vec;
        std::vector<double> p_vec;
        T_vec.resize(std::size_t(parameters["N_replica"]));
        p_vec.resize(std::size_t(parameters["N_replica"]));
        for (int i=0; i<T_vec.size(); i++){
            Tp_points >> T_vec[i];
            Tp_points >> p_vec[i];
        }
        Tp_points.close();



        auto this_temp = [&parameters, &T_vec](size_t i, size_t N) -> temperature {
            return T_vec[i];
        }(comm_world.rank(), comm_world.size());




        if(!parameters.is_restored() )
            sim.reset_sweeps(!sim.update_phase_point(this_temp));            
        sim.update_phase_point(this_temp);

//        if (is_master)
//            std::cout << "Simulating " + std::to_string(N_replica) + " replica" << std::endl;

        bool finished = sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

        mpi::barrier(comm_world);


        // Checkpoint the simulation
        mpi::mutex archive_mutex(comm_world);

//            std::cout << "Checkpointing simulation to " << checkpoint_file << std::endl;
        {
            std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
//                sim.save(checkpoint_file);
        }
        sim.update_phase_point(this_temp);
                
        mpi::barrier(comm_world);
        
        alps::results_type<my_sim_type>::type results = alps::collect_results(sim);
        
        // Save results to the .out.h5 file
        {    


            //  Delivers a vector of separate OP names
            std::string orders = parameters["orders"];
            size_t pos = 0;
            std::string delimiter = ";";
            std::string token;
            std::vector<std::string> op_names;
            while ((pos = orders.find(delimiter)) != std::string::npos)
            {
                token = orders.substr(0, pos);
                op_names.push_back(token);
                orders.erase(0, pos + delimiter.length());
            }

            using alps::accumulators::result_wrapper;   // This also adds the Jackknife binning, without need of extensive timeseries analysis !

            std::lock_guard<mpi::mutex> archive_guard(archive_mutex);
            std::string output_file = parameters["outputfile"];
            alps::hdf5::archive ar(output_file, "w");
            if (comm_world.rank() == 0 )
                ar["/parameters"] << parameters;
            ar["/simulation/results/" + std::to_string(comm_world.rank())] << results;
    
    
            for (std::string op : op_names){
                const result_wrapper& var  = results[op]; 
                const result_wrapper& var2 = results[op + "^2"];
                const result_wrapper& var4 = results[op + "^4"];
                const result_wrapper& Susc = var2 - var*var;
                const result_wrapper& Kurt = var4 / ( var2*var2 );
                ar["/simulation/results/" + std::to_string(comm_world.rank()) + "/" + op +"_Susc"] << Susc;
                ar["/simulation/results/" + std::to_string(comm_world.rank()) + "/" + op +"_Kurt"] << Kurt;
            }

            auto slice_point = sim.phase_space_point();            

            ar["/simulation/results/" + std::to_string(comm_world.rank()) + "/point"]
                        << std::vector<double>{slice_point.begin(), slice_point.end()};
            std::cout << "Finished saving for core " << std::to_string(comm_world.rank()) << std::endl;

            ar.close();

        }        
        //Post Processing
        int MPI_Barrier(MPI_Comm comm_world);
        if (comm_world.rank() == 0) {
            std::cout << "All copies finished.\n";
        }

        return 0;
    } catch (const std::runtime_error& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
