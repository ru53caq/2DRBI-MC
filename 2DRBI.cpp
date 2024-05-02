#include <alps/params/convenience_params.hpp>
#include "2DRBI.hpp"

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
    , J_x(lat_sites)
    , J_y(lat_sites)
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

    simdir = "../L_" + std::to_string(L) + "/" + "p_"+std::to_string(p).substr(0,5) + "/"  + initial_state + "/Seed_0/";

    std::ifstream Tp_points(simdir + "T-p_points.data");
    int Nreps = parameters["N_replica"];
    T_vec.resize(Nreps+1);
    p_vec.resize(Nreps+1);
    time_in_Ti.resize(Nreps+1);
    for (int i=0; i<T_vec.size(); i++){
        Tp_points >> T_vec[i];
        Tp_points >> p_vec[i];
    }
    Tp_points.close();


    //PT ACCEPTANCE RATIO
    measurements() << alps::accumulators::FullBinningAccumulator<double>("Acceptance");

    // likelihood of having even boundary conditions (complementary is odd)
//    measurements() << alps::accumulators::FullBinningAccumulator<double>("is_even");

    op_names = parsing_op(orders);
    for (std::string op : op_names){
        measurements()
        << alps::accumulators::FullBinningAccumulator<double>(op)
        << alps::accumulators::FullBinningAccumulator<double>(op + "^2")
        << alps::accumulators::FullBinningAccumulator<double>(op + "^4")
        << alps::accumulators::MeanAccumulator<std::vector<double>>(op + "_Hist");
    }
}


void ising_sim::update() {
    //Loading p-value given T, importing the bond configuration and, if necessary, import spin configuration
    if (sweeps == 0){

        std::ifstream PTvalue("PTval.txt");
        PTvalue >> PTval;
        PTvalue.close();

        T=temp.temp;     //Each replica starts at a given T,p
        up = true;      
        auto it = std::find(T_vec.begin(), T_vec.end(), temp.temp);
        ind = std::distance( T_vec.begin(), it);    //ind contains the index of the current T-p point
        N_core = ind;
        
        p = p_vec[ind];
        std::string p_str = (std::to_string(p)).substr(0,5); 
        std::ifstream disorder_config(simdir + "config_p=" + p_str + ".data");
        for (int i=0; i<J_x.size(); i++){
            disorder_config >> J_x[i];
            disorder_config >> J_y[i];
        }
        disorder_config.close();

        //In case of Zratio calculation, reload the final spin configuration available
        if (PTval ==0){
            std::ifstream spin_config(std::to_string(T)+ ".data");
            for (int i=0; i<S.size(); i++)
                spin_config >> S[i];
        spin_config.close();
        }


        E_tot = total_energy();
    }
    // measuring relevant observables (useless in the current code version)
    if (sweeps % measuring_sweeps == 0) // placed before update
        record_measurement();


    double beta = 1./T;//temp.temp;

/*
    //TESTING LATTICE STRUCTURE
    for (int i = 0; i < lat_sites; ++i){
        std::cout << "site " << i <<" couples with\n" << "\t" <<lat.nb_3(i)<<",  J="<<J_x[lat.nb_3(i)] << "\n" <<  
                                                "\t" <<lat.nb_4(i)<<",  J="<<J_y[lat.nb_4(i)] << "\n" <<  
                                                "\t" <<lat.nb_1(i)<<",  J="<<J_y[i] << "\n" <<  
                                                "\t" <<lat.nb_2(i)<<",  J="<<J_x[i] << "\n\n";
    }
*/

    //Heatbath update
    for (int i = 0; i < lat_sites; ++i)
        Heatbath(random_site(rng), beta);

    //Overrelaxation update
    overrelaxation();


    //Parallel tempering
    if ( (PTval > 0 ) && ( (sweeps + 1) % pt_sweeps == 0 ) ){

        double current_energy = total_energy();

        phase_point temp_old = temp;

        std::vector<int> other_J_x(lat_sites);
        std::vector<int> other_J_y(lat_sites);

        negotiate_update(rng, true,
        [&](phase_point other) {
            auto it = std::find(T_vec.begin(), T_vec.end(), other.temp);
            int index = std::distance( T_vec.begin(), it);
            double other_p = p_vec[index];
            std::string other_p_str = (std::to_string(other_p)).substr(0,5); 
            std::ifstream conf(simdir + "config_p=" + other_p_str + ".data");
            for (int i=0; i<J_x.size(); i++){
                conf >> other_J_x[i];
                conf >> other_J_y[i];
            }
            conf.close();
            //compute energy of current spin config with the new potential bond config
            double other_energy = 0.;
            for(int i = 0; i < lat_sites - 2*(L+1)/2 ; ++i)
                other_energy += - other_J_x[i] * ( S[i] * S[lat.nb_2(i)] ) + other_J_y[i] * ( S[i] * S[lat.nb_1(i)] );
            for(int i = (L+1)*(L-1)/2 ; i < (L+1)*(L-1)/2 + (L+1)/2 ; ++i)
                other_energy += - S[i] * other_J_x[i] * S[i - ((L+1) * (L-1)) / 2];
            for(int i = (L+1)*(L-1)/2  + (L+1)/2 ; i < (L+1)*(L-1)/2 + 2*(L+1)/2 ; ++i)
                other_energy += - S[i] * other_J_x[i] * S[i - 2*(L+1)/2];

            double dE= other_energy - current_energy;
            return -(other_energy / other.temp  - current_energy / temp.temp);  
        });

        //keep track of the p-T point at which the replica is currently at
        //Once PT is over, the computation of Z_i/Z_i+1 ratio will start
        temp!= temp_old  ? (pt_checker=1) : (pt_checker=0);
        if (pt_checker ==1){
            auto it = std::find(T_vec.begin(), T_vec.end(), temp.temp);
            int index = std::distance( T_vec.begin(), it);
            double other_p = p_vec[index];
//            std::cout << "Update accepted" << std::endl;
            std::string other_p_str = (std::to_string(other_p)).substr(0,5); 
            std::ifstream conf(simdir + "config_p=" + other_p_str + ".data");
            for (int i=0; i<J_x.size(); i++){
                conf >> J_x[i];
                conf >> J_y[i];
            }
            conf.close();
            p = other_p;
            E_tot = total_energy();
            T = temp.temp;
            ind = index;
        }

    }


    //embarassingly parallel movement between the initial p-T point and its neighbour 
    if ( (PTval ==0) && ( (sweeps + 1) % pt_sweeps == 0 ) ){

        double current_energy = total_energy();
        std::vector<int> other_J_x(lat_sites);
        std::vector<int> other_J_y(lat_sites);

        double other_T;
        double other_p;

        if (up){
            time_in_Ti[ind] +=1;
            other_T = T_vec[ind+1];
            other_p = p_vec[ind+1];
        }
        else{
            time_in_Ti[ind+1] +=1;
            other_T = T_vec[ind];
            other_p = p_vec[ind];
        }
        std::string other_p_str = (std::to_string(other_p)).substr(0,5); 
        std::ifstream conf(simdir + "config_p=" + other_p_str + ".data");
        for (int i=0; i<J_x.size(); i++){
            conf >> other_J_x[i];
            conf >> other_J_y[i];
        }
        conf.close();

        //compute energy of current spin config with the new potential bond config
        double other_energy = 0.;
        for(int i = 0; i < lat_sites; ++i){
            other_energy   += J_x[i] * ( S[i] * S[lat.nb_2(i)] ) + J_y[i] * ( S[i] * S[lat.nb_1(i)] );
            if (i == (L+1)/2 -1)
                other_energy += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
            else if (i < (L+1)/2 -1 ){
                other_energy += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
                other_energy += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i +1 ];
            }
            else if (i == (L+1)*(L-2)/2)
                other_energy += S[i] * J_x[ (L+1)*(L-1)/2 + L ];
            else if (i > (L+1)*(L-2)/2){
                other_energy += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) ];
                other_energy += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) +1];
            }
        }
        other_energy = - other_energy;
        double dE= other_energy - current_energy;

        double u = std::uniform_real_distribution<double>{0., 1.}(rng);
        double alpha = std::exp(-(other_energy / other_T  - current_energy / T) );
        if(u<alpha){
            T=other_T;
            p=other_p;
            J_x = other_J_x;
            J_y = other_J_y;
            up = !up;
            E_tot = total_energy();
        }

        // Checkpoint the amount of times each p-T point has been visited
        if ( ( (sweeps + 1) % (1000*pt_sweeps) == 0 ) ){
            std::ofstream ofs("Ti_time_rep_"+std::to_string(N_core) + ".txt");
            for (auto & val : time_in_Ti)
                ofs << val << " ";
            ofs.close();
        }

        if (sweeps == thermalization_sweeps+total_sweeps -1)
            std::cout << time_in_Ti[ind]/time_in_Ti[ind+1]<< std::endl;
    }


    ++sweeps;

    //In case of Zratio calculation, reload the final spin configuration available
    if ( (sweeps% (1000*pt_sweeps) ==0)  && (sweeps>thermalization_sweeps) && (PTval == 1 )){
        std::ofstream spin_config(std::to_string(T) + ".data");
        for (int i=0; i<S.size(); i++)
            spin_config << S[i] << "\n";
    spin_config.close();
    }
}

// Collects the measurements
void ising_sim::measure() {
    if (sweeps < thermalization_sweeps && (sweeps+1) != measuring_sweeps || ((sweeps+1) % measuring_sweeps !=0))
        return;

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

    if (i == (L+1)/2 -1)
        e += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
    else if (i < (L+1)/2 -1 ){
        e += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
        e += S[i] * J_x[ (L+1) * (L-1)/2 + 2*i +1 ];
    }
    else if (i == (L+1)*(L-2)/2)
        e += S[i] * J_x[ (L+1)*(L-1)/2 + L ];
    else if (i > (L+1)*(L-2)/2){
        e += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) ];
        e += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) +1];
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
            E += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) ];
            E += S[i] * J_x[ (L+1) * (L-1)/2 +  L + 2*( i - (L+1)*(L-2)/2) +1];
        }
    }
    return - E; // normalized by # spins
}

double ising_sim::mag_FM() {
    double mag = 0.;
    for (int i = 0; i < lat_sites; i+=1)
        mag += S[i];
    return fabs(mag) / lat_sites;
}

void ising_sim::Heatbath(int i, double beta) {
    if (i%L>0 && i%L < L-1){
        double E_i = E_tot;
        double E_f = E_i;
        int S_i = S[i];
        E_f += - 2*local_energy(i);
        S[i]*= -1; 
        double u = std::uniform_real_distribution<double>{0., 1.}(rng);
        double alpha = std::exp( -beta*(E_f-E_i));    //Metropolis update 
    //    double compl_prob = 1./ (1. + alpha);                 //Heat bath  update
        if (u<alpha)
    //    if (u > compl_prob)
            E_tot=E_f;
        else
            S[i]=S_i;
    }
}

void ising_sim::overrelaxation(){
    for (int i = 0; i < lat_sites; ++i){
        if (local_energy(i)==0 &&  i%L>0 && i%L < L-1 ){
//            bool flip=std::bernoulli_distribution{0.5}(rng);
//            if (flip)
                S[i] *= -1;
        }
    }
}

void ising_sim::initialization(std::string str) {
    if (str== "Hot&Cold"){
        int m = (std::bernoulli_distribution{}(rng) ? 1 : -1);
        if (m==1){
            for (auto & v : S)
                v=1;      
        }
        else{
            for (auto & v : S)
                v = (std::bernoulli_distribution{}(rng) ? 1 : -1);   
        }
    }
    else if (str== "Cold"){
        for (auto & v : S)
            v=1;
    }
    else{
        for (auto & v : S)
            v = (std::bernoulli_distribution{}(rng) ? 1 : -1);
    }

    // Fix spins at the x-edges to be 1/1 for even BC and 1/-1 for odd BC 
    if (str == "even"){
        for (auto & v : S)
            v = (std::bernoulli_distribution{}(rng) ? 1 : -1);
        for (int i=0; i < lat_sites ; i++){
            if (i%L == 0)
                S[i] = 1;
            if (i%L == L-1)
                S[i] = 1;
        }
    }
    if (str == "odd"){
        for (auto & v : S)
            v = (std::bernoulli_distribution{}(rng) ? 1 : -1);
        for (int i=0; i < lat_sites ; i++){
            if (i%L == 0)
                S[i] = 1;
            if (i%L == L-1)
                S[i] = -1;
        }
    }
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



