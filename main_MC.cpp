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
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        if (is_master){
            std::string filename = "PTval.txt";
            std::ifstream file(simdir + filename);
        if (!file.is_open()) {
                // If the file doesn't exist, create it and write 1 inside
                std::ofstream newFile(simdir + filename);
                newFile << "1"; // Write 1 inside the file
                newFile.close();
            } 
        else 
            file.close();
       }
        
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
        std::ifstream Tp_points(simdir+"T-p_points.data");
        std::vector<double> T_vec;
        std::vector<double> p_vec;
        T_vec.resize(std::size_t((int) parameters["N_replica"]+1));
        p_vec.resize(std::size_t((int) parameters["N_replica"]+1));
        for (int i=0; i<T_vec.size(); i++){
            Tp_points >> T_vec[i];
            Tp_points >> p_vec[i];
        }
        Tp_points.close();


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

        // COMPUTE AND STORE ZRATIOS FOR EACH REPLICA
		std::vector<double> ratiovec;
		std::vector<double> ratio_uncvec;
		ratiovec.resize((int) parameters["N_replica"]);
		ratio_uncvec.resize((int) parameters["N_replica"]);
		for (int i=0;i<(int) parameters["N_replica"];i++){
            double N1;
            double N2;
			std::ifstream Ti_reps("Ti_time_rep_"+std::to_string(i)+".txt");
            Ti_reps >> N1;
            Ti_reps >> N2;
            Ti_reps.close();
			ratiovec[i] = static_cast<double>(N1)/N2;
			ratio_uncvec[i]= std::sqrt( N1/std::pow(N2,2) + std::pow(N1,2)/std::pow(N2,3) ); 
			ar["results/"+std::to_string(i)+"/N1"] << N1;
			ar["results/"+std::to_string(i)+"/N2"] << N2;
		}	
		ar["/parameters"] << parameters;
		ar["/parameters/T_points"] << T_vec;
		ar["/parameters/p_points"] << p_vec;
		ar["/results/Zratios"] << ratiovec;
		ar["/results/Zratios_unc"] << ratio_uncvec;

        // Compute Zratio product = Z_TOT
        double ratioprod = std::accumulate(ratiovec.begin(), ratiovec.end(), 1.0, std::multiplies<double>());
        double rel_unc_sq_sum = 0.;
        for (size_t i = 0; i < ratiovec.size(); ++i)
            rel_unc_sq_sum += std::pow(ratio_uncvec[i] / ratiovec[i], 2);
        double ratioprod_unc = ratioprod * std::sqrt(rel_unc_sq_sum);
        ar["/results/Z_TOT"] << ratioprod;
        ar["/results/Z_TOT_unc"] << ratioprod_unc;		
        std::cout << ratioprod << std::endl;



        // Compute and store the timeseries analysis
        std::vector<double> Z_prod_ts;
        std::vector<double> dZ_prod_ts;
        for (int i=0;i<(int) parameters["N_replica"];i++){
            std::vector<double> Z_i_ts;
            std::vector<double> dZ_i_ts;
            std::ifstream Z_i_ts_file("Z_" + std::to_string(i) + "_timeseries.txt");
            Z_i_ts.insert(Z_i_ts.end(), std::istream_iterator<double>(Z_i_ts_file), std::istream_iterator<double>());
            Z_i_ts_file.close();
            std::ifstream dZ_i_ts_file("dZ_" + std::to_string(i) + "_timeseries.txt");
            dZ_i_ts.insert(dZ_i_ts.end(), std::istream_iterator<double>(dZ_i_ts_file), std::istream_iterator<double>());
            dZ_i_ts_file.close();
            for (int j=0; j<Z_i_ts.size();j++){
                if (i==0){
                    Z_prod_ts = Z_i_ts;
                    dZ_prod_ts.resize(Z_i_ts.size(), 0.);
                }
                else
                    Z_prod_ts[j] = Z_prod_ts[j] * Z_i_ts[j];        
                dZ_prod_ts[j] += std::pow(dZ_i_ts[j] / Z_i_ts[j], 2);
            }

        }
        for (int j=0; j<Z_prod_ts.size();j++)
            dZ_prod_ts[j] = std::sqrt(dZ_prod_ts[j]) * Z_prod_ts[j];        
        ar["/results/Z_timeseries"] << Z_prod_ts;
        ar["/results/Z_unc_timeseries"] << dZ_prod_ts;



        //Store bond configurations, Temperature values and spin configurations for each simulation
		for (int n_sim=0;n_sim<(int) parameters["N_replica"];n_sim++){   
		        ar["/results/" + std::to_string(n_sim) + "/T"] << T_vec[n_sim];
		        ar["/results/" + std::to_string(n_sim) + "/p"] << p_vec[n_sim];
		        //store the bond configuratio used for each p-T value simulation
			std::ostringstream oss1;
		    	oss1 << simdir << "config_p=" << std::fixed << std::setprecision(3) << p_vec[n_sim] << ".data";
			std::ifstream bondconfigfile(oss1.str());
		        std::vector<int> bondconfig;    
			std::string line;
			while (std::getline(bondconfigfile, line)) {
		            std::istringstream iss(line);  // Use a string stream to parse each line
		            int value;
		            while (iss >> value)
		                bondconfig.push_back(value);
		        }
		        ar["/results/" + std::to_string(n_sim) + "/bonds"] << bondconfig;
		        bondconfigfile.close();

		        //store the midpoint spin configuration where Zratio sampling starts
                // THIS WILL JUST BE RANDOM IN THE NEW METHOD
		        std::ostringstream oss2;
		        oss2 << simdir << std::fixed << std::setprecision(6) << T_vec[n_sim] << ".data";
		        std::ifstream spinconfigfile(oss2.str());
		        std::vector<int> spinconfig((std::istream_iterator<int>(spinconfigfile)), std::istream_iterator<int>());
		        ar["/results/" + std::to_string(n_sim) + "/spinconfig"] << spinconfig;
		        spinconfigfile.close();
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
