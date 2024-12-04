#include <alps/params/convenience_params.hpp>
#include "2DRBI.hpp"
#include "mpi.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <chrono>

// Defines the parameters for the ising simulation
void ising_sim::define_parameters(parameters_type & parameters) {
    // If the parameters are restored, they are already defined
    if (parameters.is_restored()) {
        return;
    }
    // Adds the parameters of the base class
    Base::define_parameters(parameters);
    alps::define_convenience_parameters(parameters)
        .description("square lattice 2D RBI model simulation")                                                           
        .define<int>("L", 24, "linear size in x direction")                                                         
        // Monte Carlo parameters                                                                         
        .define<int>("total_sweeps", 0, "max sweeps (0 means indefinite); need to be specfied in tksvm")
        .define<int>("thermalization_sweeps", 1000, "number of sweeps for thermalization")
        .define<int>("measuring_sweeps", 10, "number of sweeps between measurements")
        .define<int>("pt_sweeps", 10, "number of sweeps between PT updates")
        .define<int>("N_replica", 1, "Number of replicas simulated")
        .define<double>("disorder", 1, "Probability of J_<i,j> and K_<i,j> being inverted(0.5 for maximal disorder)")
        .define<int>("N_avg_p", 1, "Number of different realizations of the disordered system")        
        .define<int>("NBins", 1000, "Number of bins used by energy histogram")
        .define<double>("sampling_range_a", 0, "sampling control; lower bound")
        .define<double>("sampling_range_b", 0.1, "sampling control; upper bound")
        .define<std::string>("orders", "Energy;", "orders to measure, the ';' needed")
        .define<std::string>("initial_state", "Random", "determine initial configuration");
    phase_space_point::temperature::define_parameters(parameters);
}

// Creates a new simulation.
// We always need the parameters and the seed as we need to pass it to
// the alps::mcbase constructor. We also initialize our internal state,
// mainly using values from the parameters.
ising_sim::ising_sim(parameters_type & parms, std::size_t seed_offset)
    : Base(parms, seed_offset)
    , rng(parameters["SEED"].as<std::size_t>() + seed_offset)
    , rng2(parameters["SEED"].as<std::size_t>() + seed_offset)
    , L(parameters["L"])
    , lat_sites( ((L+1) * (L-1)) / 2 )         // Given TC in rotated formalism of size L, the lattice of interest for the RBI takes this form
    , lat(L)
    , op_measure(L)
    , sweeps(0) //current sweep
    , total_sweeps(parameters["total_sweeps"])
    , thermalization_sweeps(parameters["thermalization_sweeps"])
    , measuring_sweeps(parameters["measuring_sweeps"])
    , pt_sweeps(parameters["pt_sweeps"])
    , temp(parameters)
    , N_replica(5)
    , p(parameters["disorder"].as<double>())
    , J_x(lat_sites + 2*L)
    , J_y(lat_sites + 2*L)
    , NBins(parameters["NBins"])
    , sampling_range_a(parameters["sampling_range_a"].as<double>())
    , sampling_range_b(parameters["sampling_range_b"].as<double>())
    , random_site(0, lat_sites - 1)
    , rand_L(0, L-1)
    , orders(parameters["orders"].as<std::string>())
    , initial_state(parameters["initial_state"].as<std::string>())
{
    

    S.resize(std::size_t(lat_sites));

    initialization(initial_state);

    simdir = "./";

    std::ifstream Tp_points(simdir + "T-p_points.data");
    N_replica = parameters["N_replica"];
    T_vec.resize(N_replica+1);
    p_vec.resize(N_replica+1);
    time_in_Ti.resize(2);
    for (int i=0; i<T_vec.size(); i++){
        Tp_points >> T_vec[i];
        Tp_points >> p_vec[i];
    }
    Tp_points.close();

//  Prepare the J configs for this replica and the next; will speed up Zratio swaps
    this_J_x.resize(lat_sites+2*L);
    this_J_y.resize(lat_sites+2*L);
    next_J_x.resize(lat_sites+2*L);
    next_J_y.resize(lat_sites+2*L);

/*
    //PT ACCEPTANCE RATIO [NOT ACTUALLY USED]
    measurements() << alps::accumulators::FullBinningAccumulator<double>("Acceptance");
    op_names = parsing_op(orders);
    for (std::string op : op_names){
        measurements()
        << alps::accumulators::FullBinningAccumulator<double>(op)
        << alps::accumulators::FullBinningAccumulator<double>(op + "^2")
        << alps::accumulators::FullBinningAccumulator<double>(op + "^4")
        << alps::accumulators::MeanAccumulator<std::vector<double>>(op + "_Hist");
    }
*/
}


void ising_sim::update() {
    if (sweeps == 0){
//      Each replica starts at a given T,p (T,p) and (T+1,p+1) with respective J_x,J_y configs
//      Simulation switches between UP= true to say we are in (T,p), up=False to say we are in (T+1,p+1)
        up = true;      

        MPI_Comm_rank(MPI_COMM_WORLD, &N_core);
        ind = N_core;
        this_p = p_vec[ind];
        this_T = T_vec[ind];
        next_p = p_vec[ind+1];
        next_T = T_vec[ind+1];

        T=temp.temp;     
        p = p_vec[ind];

        std::ostringstream out;
        out << std::fixed << std::setprecision(3) << p;
        std::string p_str = out.str(); 
        std::ifstream disorder_config(simdir + "config_p=" + p_str + ".data");
        for (int i=0; i<J_x.size(); i++){
            disorder_config >> J_x[i];
            disorder_config >> J_y[i];
            this_J_x[i] = J_x[i];
    	    this_J_y[i] = J_y[i];
    	}
        disorder_config.close();

        std::ostringstream next_out;
        next_out << std::fixed << std::setprecision(3) << next_p;
        std::string next_p_str =next_out.str(); 
        std::ifstream next_disorder_config(simdir + "config_p=" + next_p_str + ".data");
        for (int i=0; i<J_x.size(); i++){
            next_disorder_config >> next_J_x[i];
            next_disorder_config >> next_J_y[i];
        }
        next_disorder_config.close();

	for (int i = 0; i < J_x.size(); i++) {
            if (this_J_x[i] != next_J_x[i])
    		    diff_J_x.push_back(i);
            if (this_J_y[i] != next_J_y[i])                
                    diff_J_y.push_back(i);
	}
    E_tot = total_energy();

   }

    double beta = 1./T;

    //Heatbath update
    for (int i = 0; i < 10; ++i)
        Heatbath(random_site(rng), beta);
    //Overrelaxation update
//    overrelaxation();         

    //embarassingly parallel movement between the initial p-T point and its neighbour 
    if ( (sweeps) % pt_sweeps == 0 ){

        double current_energy = E_tot;
        std::vector<int> other_J_x(lat_sites+2*L), other_J_y(lat_sites+2*L);
        double other_T, other_p;
        if (up){
            if (sweeps>thermalization_sweeps)
        	   time_in_Ti[0] ++;     //actually used in computations
            n1 ++;                   //only for timeseries analysis
    	    other_T = next_T;
    	    other_p = next_p;
    	    other_J_x = next_J_x;
    	    other_J_y = next_J_y;
   	    }
        else{
            if (sweeps>thermalization_sweeps)	
	    	  time_in_Ti[1] ++;     //actually used in computations
            n2++;                   // only for timeseries analysis
    	    other_T = this_T;
    	    other_p = this_p;
    	    other_J_x = this_J_x;
    	    other_J_y = this_J_y;
        }

        double deltaE = 0.;
        for(int i = 0; i < diff_J_x.size(); i++){
   	        if (diff_J_x[i] < lat_sites)
	           deltaE += J_x[ diff_J_x[i] ] * (S[ diff_J_x[i] ] * S[lat.nb_2( diff_J_x[i] )]);
            else if ( diff_J_x[i]< lat_sites+L)
                deltaE += S[ (int)( ( diff_J_x[i] - lat_sites - (diff_J_x[i] - lat_sites)%2)/2 )  ] * J_x[ diff_J_x[i]];
            else if (diff_J_x[i]>= lat_sites+L)
                deltaE += S[ (int)( ( diff_J_x[i] - lat_sites - L +1  - (diff_J_x[i] - lat_sites - L + 1 )%2)/2 + (L+1)*(L-2)/2 )  ] * J_x[ diff_J_x[i]];
        }
    	for(int i = 0; i < diff_J_y.size(); i++)
	       deltaE += J_y[ diff_J_y[i] ] * (S[ diff_J_y[i] ] * S[lat.nb_1( diff_J_y[i] )]);

    	double other_energy = current_energy + 2.*deltaE;  
        double u = std::uniform_real_distribution<double>{0., 1.}(rng);
        double alpha = std::exp(-(other_energy / other_T  - current_energy / T) );
        if(u<alpha){
            T=other_T;
            p=other_p;
            J_x = other_J_x;
            J_y = other_J_y;
            up = !up;
            E_tot = other_energy;
        }


        // Timeseries of Zratios 
        if ( (sweeps>thermalization_sweeps+1)&& (sweeps%(1000*pt_sweeps) == 0) ){
            n1 = time_in_Ti[0];
            n2 = time_in_Ti[1];
            double Z_r = n1/(double)n2;
            double dZ_r = std::sqrt( ((double)n1)/std::pow(n2,2) + ((double)(std::pow(n1,2)))/std::pow(n2,3) );
            Z_i.push_back(Z_r);
            dZ_i.push_back(dZ_r);
        }

        // Checkpoint the amount of times each p-T point has been visited, Z_i and dZ_i timeseries
        if ( sweeps == thermalization_sweeps+total_sweeps - 10 ){
            std::ofstream ofs1("Ti_time_rep_"+std::to_string(N_core) + ".txt");
            for (auto & val : time_in_Ti)
                ofs1 << val << " ";
            ofs1.close();
            std::ofstream ofs2("Z_"+std::to_string(N_core) + "_timeseries.txt");
            for (auto & val : Z_i)
                ofs2 << val << " ";
            ofs2.close();
            std::ofstream ofs3("dZ_"+std::to_string(N_core) + "_timeseries.txt");
            for (auto & val : dZ_i)
                ofs3 << val << " ";
            ofs3.close();
        }
    }

    ++sweeps;
}

// Collects the measurements; NOT NEEDED IN OUR MINIMAL CASE
void ising_sim::measure() {
//    if (sweeps < thermalization_sweeps && (sweeps+1) != measuring_sweeps || ((sweeps+1) % measuring_sweeps !=0))
    if (sweeps > -5)
        return;
/*
    Nmeas +=1;
    measurements()["Acceptance"] << pt_checker;

//    measurements()["is_even"] << ((S[0]*S[L-1] +1)/2);
    for (std::string op : op_names){     

        double op_value;
        std::vector<double> OP_Hist;
        OP_Hist.resize(std::size_t(NBins + 1));
        std::fill(OP_Hist.begin(),OP_Hist.end(),0);
        
        if (op == "Energy"){
            op_value = total_energy()/lat_sites;
            if (op_value<=0)
                OP_Hist[std::size_t( (-1.* op_value + 1e-9 )/ (2./(NBins)))] += 1;
            if (op_value>0)
                OP_Hist[0] += 1e-9;
        }
        else if (op == "mag_FM") {
            op_value = mag_FM();
            if (op_value>=0)
                OP_Hist[std::size_t( (op_value + 1e-9)/ (1./(NBins)))] += 1;
            if (op_value<0)
                OP_Hist[0] += 1e-9;
        }
        measurements()[op] << op_value;
        measurements()[op + "^2"] << op_value * op_value;
        measurements()[op + "^4"] << op_value * op_value* op_value * op_value;
        measurements()[op + "_Hist"] << OP_Hist;
    }
*/
}

// Returns a number between 0.0 and 1.0 with the completion percentage
double ising_sim::fraction_completed() const {
    double f=0.;
    if (total_sweeps>0 && sweeps >= thermalization_sweeps) 
        f=(sweeps-thermalization_sweeps)/double(total_sweeps);
    if (sweeps> thermalization_sweeps + total_sweeps){
        f=1.;
    }
    return f;
}

ising_sim::phase_point ising_sim::phase_space_point() const {
    return temp;
}

bool ising_sim::update_phase_point(phase_point const& pp) {
    Base::update_phase_point(pp);
    bool changed = (pp != temp);
    if (changed) {
        temp = pp;
    }
    return changed;
}

// Saves the state to the hdf5 file
void ising_sim::save(alps::hdf5::archive & ar) const {

    Base::save(ar);
    ar["checkpoint/sweeps"] << sweeps;

    // save phase point (not directly interaction parameters)
    ar["checkpoint/temp"] << temp.temp;
    ar["checkpoint/J_x"] << J_x;
    ar["checkpoint/J_y"] << J_y;

    // save current configuration
    std::vector<int> spins;
    for(auto const& v : S)
        spins.insert(spins.end(), v);
    ar["checkpoint/configuration"] << spins;

    // random number engine
    std::ostringstream engine_ss;
    engine_ss << rng;
    ar["checkpoint/random"] << engine_ss.str();

    //save measurement related quantities 
    ar["checkpoint/Nmeas"] << Nmeas;
    ar["checkpoint/E_tot"] << E_tot;

    std::cout << "checkpointing T= " << temp.temp << " " << " at sweep " << sweeps << std::endl;

}

void ising_sim::load(alps::hdf5::archive & ar) {
    Base::load(ar);
    thermalization_sweeps = parameters["thermalization_sweeps"];
    // Note: `total_sweeps` is not restored here!
    ar["checkpoint/sweeps"] >> sweeps;

    // restore phase point (not directly interaction parameters)
    ar["checkpoint/temp"] >> temp.temp;
    ar["checkpoint/J_x"] >> J_x;
    ar["checkpoint/J_y"] >> J_y;

    // restore spin configurations
    std::vector<int> spins;
    ar["checkpoint/configuration"] >> spins;
    for(int i = 0; i < lat_sites; ++i)
        S[i] = spins[i];

    // random number engine
    std::string engine_str;
    ar["checkpoint/random"] >> engine_str;
    std::istringstream engine_ss(engine_str);
    engine_ss >> rng;
    //load measurement related quantities 
    ar["checkpoint/Nmeas"] >> Nmeas;
    ar["checkpoint/E_tot"] >> E_tot;

    E_tot = total_energy();
}


double ising_sim::local_energy(int i){
    double e = S[i] * (J_x[i]*S[lat.nb_2(i)] + J_x[lat.nb_3(i)]*S[lat.nb_3(i)] + J_y[i]*S[lat.nb_1(i)] + J_y[lat.nb_4(i)]*S[lat.nb_4(i)]);
    if (i == (L+1)/2 -1){
        e += S[i] * J_x[ (int)( (L+1) * (L-1)/2 + 2*i ) ];
    }
    else if (i < (L+1)/2 -1 ){
        e += S[i] * J_x[ (int)((L+1) * (L-1)/2 + 2*i )];
        e += S[i] * J_x[ (int)((L+1) * (L-1)/2 + 2*i +1 )];
    }
    else if (i == (L+1)*(L-2)/2){
        e += S[i] * J_x[ (int)((L+1)*(L-1)/2 + L )];
    }
    else if (i > (L+1)*(L-2)/2){
        e += S[i] * J_x[ (int)((L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) -1)];
        e += S[i] * J_x[ (int)((L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) )];
    }
    return - e;
}


double ising_sim::total_energy(){
    double E = 0.;
    for(int i = 0; i < lat_sites; ++i){
        E   += J_x[i] * ( S[i] * S[lat.nb_2(i)] ) + J_y[i] * ( S[i] * S[lat.nb_1(i)] );
        if (i == (L+1)/2 -1)
            E += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
        else if (i < (L+1)/2 -1 ){
            E += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
            E += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i +1 ];
        }
        else if (i == (L+1)*(L-2)/2)
            E += S[i] * J_x[ (L+1)*(L-1)/2 + L ];
        else if (i > (L+1)*(L-2)/2){
            E += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) -1];
            E += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) ];
        }
    }
    return - E;
}

double ising_sim::mag_FM() {
    double mag = 0.;
    for (int i = 0; i < lat_sites; i+=1)
        mag += S[i];
    return fabs(mag) / lat_sites;
}

void ising_sim::Heatbath(int i, double beta) {
    double E_i = E_tot;
    double E_f = E_i;
    int S_i = S[i];
    E_f += - 2 * local_energy(i);
    S[i]*= -1;
    double u = std::uniform_real_distribution<double>{0., 1.}(rng);
    double alpha = std::exp( -beta*(E_f-E_i));    //Metropolis update 
    double compl_prob = 1./ (1. + alpha);                 //Heat bath  update
//    if (u<alpha)
    if (u > compl_prob)
        E_tot = E_f;
    else
        S[i] = S_i;
}

void ising_sim::overrelaxation(){
    for (int i = 0; i < lat_sites; ++i){
        if (local_energy(i)==0 &&  i%L>0 && i%L < L-1 ){
                S[i] *= -1;
        }
    }
}

void ising_sim::initialization(std::string str) {
    for (auto & v : S)
        v = (std::bernoulli_distribution{}(rng) ? 1 : -1);
}

std::vector<std::string> ising_sim::parsing_op(std::string str) {
    size_t pos = 0;
    std::string delimiter = ";";
    std::string token;
    std::vector<std::string> op_parsed;

    while ((pos = str.find(delimiter)) != std::string::npos)
    {
        token = str.substr(0, pos); //from current begin to the next ';'
        op_parsed.push_back(token);
        str.erase(0, pos + delimiter.length());
    }
    return op_parsed;
}

void ising_sim::reset_sweeps(bool skip_therm) {
    Base::reset_sweeps(skip_therm);
    if (skip_therm)
        sweeps = sweeps;//thermalization_sweeps;
    else
        sweeps = 0;
}

void ising_sim::record_measurement() {
    // Mmke sure ONLY store ONE temp, to avoid writting confliction
    if ( sampling_range_a <= temp.temp && temp.temp <= sampling_range_b)
    {
        std::ofstream ofs;
        ofs.open("record_p=" + std::to_string(p) + "_" + initial_state + ".data", std::ofstream::app); // append
        if (sweeps <= 1)
            ofs << "#1 temp; #2 sweeps; #3 Energy; #4 mag;\n";
        ofs << temp.temp << "\t" << sweeps << "\t"
            << total_energy() << "\t"
            << mag_FM() << "\n";     
        ofs.close();
    }
    else return;
}




