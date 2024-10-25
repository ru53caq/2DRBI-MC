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
#include <time.h>
#include <numeric>

#include <alps/hdf5.hpp>
#include <alps/params.hpp>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

int main(int argc, char** argv)
{
    double t1 = (double)clock();

    // Define the type for the simulation
    typedef ising_sim my_sim_type;
    using temperature = phase_space_point::temperature;    
    mpi::environment env(argc, argv, mpi::environment::threading::multiple);
    mpi::communicator comm_world;
    const bool is_master = (comm_world.rank() == 0);
    try {

        std::string simdir;
        simdir = "./";
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

        // SET SIMULATION PARAMETERS
        int L = parameters["L"];
        double dis = parameters["disorder"];
        std::string init_state = parameters["initial_state"];
        std::ifstream T_points(simdir+"T_points.data");
        std::vector<double> T_vec;
        T_vec.resize(std::size_t((int) parameters["N_replica"]+1));
        for (int i=0; i<T_vec.size(); i++)
            T_points >> T_vec[i];
        T_points.close();


        // SETTING TEMPERATURE POINTS OF EACH REPLICA
        auto this_temp = [&parameters, &T_vec](size_t i, size_t N) -> temperature {
            return T_vec[i];
        }(comm_world.rank(), comm_world.size());

        // RESTORE PREVIOUS SIMULATION PARAMETERS (JUST IN CASE WE RELOAD)
        if(!parameters.is_restored() )
            sim.reset_sweeps(!sim.update_phase_point(this_temp));            
        sim.update_phase_point(this_temp);

        //Running simulation
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

	if (is_master){
		alps::hdf5::archive ar("out.h5", "w");
        //Store bond configurations, Temperature values and spin configurations for each simulation

        double Z;
        double dZ;
        double N1_tot=0.;
        double N2_tot=0.;
        // COMPUTE AND STORE ZRATIOS FOR EACH REPLICA
		for (int i=0;i<(int) parameters["N_replica"];i++){
            double N1=0.;
            double N2=0.;
			std::ifstream Ti_reps("Ti_time_rep_"+std::to_string(i)+".txt");
            Ti_reps >> N1;
            Ti_reps >> N2;
            Ti_reps.close();
            N1_tot +=  N1;
            N2_tot +=  N2;
		}	

        Z = static_cast<double>(N1_tot)/N2_tot;
        dZ = std::sqrt( N1_tot/std::pow(N2_tot,2) + std::pow(N1_tot,2)/std::pow(N2_tot,3) ); 
		ar["/parameters"] << parameters;
		ar["/parameters/T_points"] << T_vec;
        ar["results/N1"] << N1_tot;
        ar["results/N2"] << N2_tot;
        ar["/results/Z_TOT"] << Z;
        ar["/results/Z_TOT_unc"] << dZ;
        


        // Compute and store the timeseries analysis
        std::vector<bool> n_ts;
        std::vector<double> t_ts;
        for (int i=0;i<(int) parameters["N_replica"];i++){
            std::vector<bool> n_i_ts;
            std::vector<double> t_i_ts;
            std::ifstream n_i_ts_file(simdir+"Timeseries_" + std::to_string(i) + ".txt");
            n_i_ts.insert(n_i_ts.end(), std::istream_iterator<double>(n_i_ts_file), std::istream_iterator<double>());
            n_i_ts_file.close();
            std::ifstream t_i_ts_file(simdir+"Timestamps_" + std::to_string(i) + ".txt");
            t_i_ts.insert(t_i_ts.end(), std::istream_iterator<double>(t_i_ts_file), std::istream_iterator<double>());
            t_i_ts_file.close();
            n_i_ts.erase(n_i_ts.begin());
            t_i_ts.erase(t_i_ts.begin());
            n_ts.insert(n_ts.end(), n_i_ts.begin(), n_i_ts.end());
            t_ts.insert(t_ts.end(), t_i_ts.begin(), t_i_ts.end());
        }
        //sorting the timeseries
        std::vector<size_t> indices(t_ts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&t_ts](size_t i1, size_t i2) {
            return t_ts[i1] < t_ts[i2];
        });
        std::vector<double> sorted_t_ts(t_ts.size());
        std::vector<bool> sorted_n_ts(n_ts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_t_ts[i] = t_ts[indices[i]];
            sorted_n_ts[i] = n_ts[indices[i]]; 
        }
        t_ts = sorted_t_ts;
        n_ts = sorted_n_ts;

        int n_bins = 1;
        while (std::pow(n_bins, 4) <= n_ts.size()) {
            ++n_bins;
        }
        std::vector<int> N1_ts_bins(n_bins, 0);
        std::vector<int> N2_ts_bins(n_bins, 0);

        for (int i = 1; i <= n_bins; ++i) {
            int start = std::pow(i - 1, 4);
            int end = std::pow(i, 4) - 1;

            for (int j = start; j <= end && j < n_ts.size(); ++j) {
                if (n_ts[j] == 0)
                    N1_ts_bins[i - 1]++;
                else
                    N2_ts_bins[i - 1]++;
            }
            if (N2_ts_bins[i-1] == 0){
                N2_ts_bins[i-1] +=1;
            }

        }
        std::vector<double> Z_ts(n_bins, 0);
        std::vector<double> dZ_ts(n_bins, 0);
        for (int i = 0; i < n_bins; ++i) {
            Z_ts[i] = N1_ts_bins[i]/(double)(N2_ts_bins[i]);
            dZ_ts[i] = std::sqrt( N1_ts_bins[i]/(double)(std::pow(N2_ts_bins[i],2)) + std::pow(N1_ts_bins[i],2)/(double)(std::pow(N2_ts_bins[2],3)) ); 
        }
        ar["/results/Z_timeseries"] << Z_ts;
        ar["/results/Z_unc_timeseries"] << dZ_ts;

        
        for (int n_sim=0;n_sim<(int) parameters["N_replica"];n_sim++){   
                ar["/results/" + std::to_string(n_sim) + "/T"] << T_vec[n_sim];
        }

		ar.close();	
	}
/*        
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
*/

        int MPI_Barrier(MPI_Comm comm_world);
        if (comm_world.rank() == 0) {
            double dt =  ( (double) clock() - t1 ) / CLOCKS_PER_SEC;
            std::cout << "All copies finished in " << dt << " s " << std::endl;
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
