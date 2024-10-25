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

    std::ifstream T_points(simdir + "T_points.data");
    N_replica = parameters["N_replica"];
    T_vec.resize(N_replica+1);
    time_in_Ti.resize(2);
    for (int i=0; i<T_vec.size(); i++){
        T_points >> T_vec[i];
    }
    T_points.close();

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
}


void ising_sim::update() {
   if (sweeps == 0){
//      Each replica starts at a given T and loads J_x,J_y config
//      Simulation switches between UP= true to say we are in the original config, up=False to say we added a line flip
//      Each core follows the CONFIG, not the Temperature!!!
//      We will add 2 parameters: N1 for UP=True (line not flipped), N2 for UP=false (line flipped)
//      Line updates are proposed every sweep!!!
//      N1 and N2 are increased IF REPLICA IS AT Tmin AND HAS THE LINE FLIPPED
        auto it = std::find(T_vec.begin(), T_vec.end(), temp.temp);
        ind = std::distance( T_vec.begin(), it);    //ind contains the index of the current T-p point
        N_core = ind;
        up = true;      //All start with no line flip
        E_tot = total_energy();
        std::ifstream disorder_config(simdir + "config_p.data");
        for (int i=0; i<J_x.size(); i++){
            disorder_config >> J_x[i];
            disorder_config >> J_y[i];
        }
   }
//    if (sweeps % measuring_sweeps == 0) // placed before update
//        record_measurement();

    double beta = 1./temp.temp;
    //Heatbath update
    for (int i = 0; i < lat_sites; ++i)
        Heatbath(random_site(rng), beta);
    //Overrelaxation update
    overrelaxation();


    //line update
    double dE = 0.;
    for (int i = 0; i < (L+1)/2; i++){
        if (i == (L+1)/2 - 1)
            dE +=  S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
        else if (i < (L+1)/2 -1 ){
            dE +=  S[i] * J_x[ (L+1) * (L-1)/2 + 2*i ];
            dE +=  S[i] * J_x[ (L+1) * (L-1)/2 + 2*i +1 ];
        }
    }
    dE = 2.*dE;
    double u = std::uniform_real_distribution<double>{0., 1.}(rng);
    double alpha = std::exp( -beta*dE);    //Metropolis update 
    double compl_prob = 1./ (1. + alpha);    //Heat bath  update
//    if (u<alpha){
    if (u > compl_prob){
        
        E_tot = E_tot + dE;
        for (int i = (L+1)*(L-1)/2; i < L + (L+1)*(L-1)/2; i++)
            J_x[i] *=-1;
        up = !up;   // This replica has a flipped line of bonds!!!
    }


    //NO USE OF PT IS EVER REQUIRE IF WE SKIP THE THERMALIZING PROCEDURE
    //Parallel tempering
    if ( sweeps % pt_sweeps == 0 ){

        double current_energy = E_tot;
        phase_point temp_old = temp;
        negotiate_update(rng, true,
        [&](phase_point other) {
            return - current_energy * ( 1./ other.temp  -  1./ temp.temp);  
        });
        temp!= temp_old  ? (pt_checker=1) : (pt_checker=0);
    }
    
    // For each replica, check if you are at T_Nishimori and update the respective element; 
    if ( ( sweeps % pt_sweeps == 0 ) && (temp.temp == T_vec[0]) ){
        if (sweeps>thermalization_sweeps)
            time_in_Ti[!up] +=1;
        if (up)
            n1+=1;
        else
            n2+=1;

        timeseries.push_back(!up);

        const auto& reference_time = TimeReference::reference_time;
        TimeReference::initReferenceTime();
        double us_ref = static_cast<double>( std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - TimeReference::reference_time).count() );
        if (timestamps.size()>0)
            timestamps.push_back(us_ref/1000 - timestamps[0]);
        else
            timestamps.push_back(us_ref/1000);
/*
        if (sweeps == sweep_t_step ){
            if (n2==0)
                n2+=1;  //in case no samples went to n2 (typically only at start of equilibration)
            double Z_r = n1/(double)n2;
            double dZ_r = std::sqrt( ((double)n1)/std::pow(n2,2) + ((double)(std::pow(n1,2)))/std::pow(n2,3) );
            Z_i.push_back(Z_r);
            dZ_i.push_back(dZ_r);

            n1=0;
            n2=0;
            t_step+=1;
            sweep_t_step = std::pow(t_step,4);
        }
*/

        if ( (sweeps > thermalization_sweeps + total_sweeps/2) && ( sweeps % 1000 == 0 ) ){
            std::ofstream ofs1(simdir+ "Ti_time_rep_"+std::to_string(N_core) + ".txt");
            for (auto & val : time_in_Ti)
                ofs1 << val << " ";
            ofs1.close();
            std::ofstream ofs2(simdir+"Timeseries_"+std::to_string(N_core) + ".txt");
            for (bool val : timeseries)
                ofs2 << val << " ";
            ofs2.close();
            std::ofstream ofs3(simdir+"Timestamps_"+std::to_string(N_core) + ".txt");
            ofs3.precision(10);
            for (auto & val : timestamps)
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
/*
    if ( (sweeps == thermalization_sweeps + total_sweeps) && (temp.temp == T_vec[0]) ){
        f=1.;
        std::ofstream ofs(simdir+"DONE.txt");
        ofs << 1;
        ofs.close();
        std::cout << N_core << " " << temp.temp << std::endl;
    }
    if ( sweeps> thermalization_sweeps + total_sweeps){
        int done;
        std::ifstream donefile(simdir+ "DONE.txt");
        if (donefile.is_open()){ 
            donefile >> done;
            donefile.close();
        }
        if (done==1){
            f=1.;
            std::cout << N_core << " " << temp.temp << std::endl;

        }
    }
*/
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
//            bool flip=std::bernoulli_distribution{0.5}(rng);
//            if (flip)
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



