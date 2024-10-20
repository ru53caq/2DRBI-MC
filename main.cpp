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

            std::cout << "Creating simulations..." << std::endl;
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



        auto this_temp = [&parameters, &T_vec](size_t i, size_t N) -> temperature {
            return T_vec[i];
        }(comm_world.rank(), comm_world.size());




        if(!parameters.is_restored() )
            sim.reset_sweeps(!sim.update_phase_point(this_temp));            
        sim.update_phase_point(this_temp);


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
		std::vector<double> ratiovec;
		std::vector<double> ratio_uncvec;
		ratiovec.resize((int) parameters["N_replica"]);
		ratio_uncvec.resize((int) parameters["N_replica"]);
		for (int i=0;i<(int) parameters["N_replica"];i++){
                        std::ostringstream oss;
                        oss << "Ti_time_rep_" << std::to_string(i) << ".txt";
			std::ifstream Ti_reps("Ti_time_rep_"+std::to_string(i)+".txt");
			std::vector<int> vec;
			vec.resize((int) parameters["N_replica"]+1);
//			std::cout << "Ti_time_rep_"+std::to_string(i)+".txt" << std::endl;
			for (int j=0; j<(int) parameters["N_replica"]+1; j++){
				Ti_reps >> vec[j];
//				std::cout << vec[j] << std::endl;
			}
			int N_1=vec[i];
			int N_2=vec[i+1];
			ratiovec[i] = static_cast<double>(N_1)/N_2;
//			std::cout << N_1 << " " << N_2 << " " << ratiovec[i] << std::endl;
			ratio_uncvec[i]= std::sqrt( N_1/std::pow(N_2,2) + std::pow(N_1,2)/std::pow(N_2,3) ); 
			Ti_reps.close();
			ar["results/"+std::to_string(i)+"/N1"] << N_1;
			ar["results/"+std::to_string(i)+"/N2"] << N_2;
			ar["results/"+std::to_string(i)+"/Zratio"] << ratiovec[i];
			ar["results/"+std::to_string(i)+"/Zratio_unc"] << ratio_uncvec[i];
//                        std::remove(oss.str().c_str());
		}	
/*
	        std::ofstream Zratios;
		Zratios.open("Zratios.txt");
        	for (auto & val : ratiovec)
	            Zratios << val << " ";
		Zratios << "\n";
		for (auto & val : ratio_uncvec)
		    Zratios << val << "  ";
	        Zratios.close();
*/	
		ar["/parameters"] << parameters;
		ar["/parameters/T_points"] << T_vec;
		ar["/parameters/p_points"] << p_vec;
		ar["/results/Zratios"] << ratiovec;
		ar["/results/Zratios_unc"] << ratio_uncvec;

                double ratioprod = std::accumulate(ratiovec.begin(), ratiovec.end(), 1.0, std::multiplies<double>());
                double rel_unc_sq_sum = 0.;
                for (size_t i = 0; i < ratiovec.size(); ++i)
                    rel_unc_sq_sum += std::pow(ratio_uncvec[i] / ratiovec[i], 2);
                double ratioprod_unc = ratioprod * std::sqrt(rel_unc_sq_sum);
                ar["/results/Z_TOT"] << ratioprod;
                ar["/results/Z_TOT_unc"] << ratioprod_unc;

		
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
		        std::ostringstream oss2;
		        oss2 << simdir << std::fixed << std::setprecision(6) << T_vec[n_sim] << ".data";
		        std::ifstream spinconfigfile(oss2.str());
		        std::vector<int> spinconfig((std::istream_iterator<int>(spinconfigfile)), std::istream_iterator<int>());
		        ar["/results/" + std::to_string(n_sim) + "/spinconfig"] << spinconfig;
		        spinconfigfile.close();
			
//			std::remove(oss1.str().c_str());
//			std::remove(oss2.str().c_str());
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
