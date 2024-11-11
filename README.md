# 2DRBI-MC
MC code for computing the free energy ratio of the 2DRBI model for varying points in p-T space.

v0.1
Code for computing the free energy ratio of the 2DRBI model for varying points in p-T space.

### Building and installation
To generate jobs and run the code, one first needs to have installed the ALPSCode libraries. Detailed instructions on how to build ALPSCore can be fournd in the
project's wiki.

The executable client code is compiled through the use of [`CMakeLists.txt`](./CMakeLists.txt). Inside  a build folder, execute the command 
```bash
$ cmake .. -DCONFIG_MAPPING=LAZY -DALPSCore_DIR=~/path_to_ALPSCore -DEigen3_DIR=~path_to_eigen3/eigen3
$ make -j8
```

## Simulation methods

# PT (Daniel_Loss path in temperature)
- Given a disorder configuration and a distribution of temperatures, run a Monte Carlo chain where different replicas are allowed to exchange Temperature label ergodically.
Highly reliant on the choice of temperature distribution.
- Line updpates between even and odd boundary conditions are proposed at all T, but will only be accepted at high temperatures. Swaps will allow percolation of even/odd boundary configurations at low T
- At the end, computes Z_even/Z_odd as the ratio of times the system remains in one or the other configuration

The code requires communication between replicas and therefore requires openMPI. 

# MC (partitioned Zratio Calculation)
calculate Z_even/Z_odd by passing through the 0-disorder axis
Simulation is divided in four sections:
-	simulation	of Z_even: N replicas are generated starting from a maximally disordered configuration; the other configurations are obtained by removing 
		disorder bonds until no error is present. For each replica, run a MC simulation proposing adjacent disorder bond configurations (dE~cost). This gives an individual Z_ratio.
 	 At the end of the simulation, take the product of Zratios and expand uncertainties to obtain Z_1, dZ_1 
		between the disordered and the 0-disorder configuration with even boundary conditions. We will call this output Z_even
- simulation of Z_odd: Same as above, but all disorder configurations have an extra line of flipped bonds. 
		After taking product of Zratios, will give Z_2, dZ_2  between the disordered and the 0-disorder configuration with odd boundary conditions. 
		We will call this output Z_odd.
- Run another simulation connecting the two configurations with even and odd BC with zero disorder, always using several replicas to ensure the probability 
		of an exchange between configurations being accepted is dE~cost . THIS NEEDS TO BE DONE ONLY ONCE PER SYSTEM SIZE FOR A FIXED TEMPERATURE T_0dis. 
		We will call this output Z_0dis.
- Compute Z = Z_even* Z_0dis / Z_odd

The code is embarassingly parallelized, meaning it will run the same computation on different cores without communication. 
	
## Workflow of the code

Numerical implementation of the MC simulation requires the generation of a set of initial configurations and simulation parameters. 

Both MC paths in disorder and PT (or Daniel_Loss) path in temperature can be run.

First, Simulations are run by submitting [`subjobs_MC.sub`](./subjobs_MC.sub). 
For each disorder iteration, the code will:
1. Generate its respective directory
2. Call [`genconfigs_MC.py`](./genconfigs_MC.py), which will generate an initialization file following the structure given by [`RBI_template.ini`](./RBI_template.ini).
3. Run a simulation for the MC method with even BC
4. Run a simulation for the MC method with odd BC
5. Run a simulation for the PT method ( if flagged)

At the end of it all, the code will run a 0disorder simulation to compute Z_0dis for the MC method (if flagged)

Then, all results of the simulations are wrapped by [`datacollect.py`](./datacollect.py)
## Code sections

### SIGNIFICANT PARTS (sections which require changes depending on parameter choice, code updates etc):

### submit_jobs.py
Generates a sequence of simulations for a given choice of parameters. Here, one needs to choose:

$ Lattice size                          L
$ a list with N temperature values	    T_vec
$ a list with N disorder values		    p_vec
$ Number of thermalization sweeps		therm
$ Number of simulation sweeps		    totsweeps
$ Number of different configurations	N_disorder_reps
$ start configuration (even or odd)	    start_config

For a given choice of initial configurations, the code generates $N_disorder_reps$ folders and .ini files, each with its own seed and set of bond configurations.

To generate the bond configurations, in each disorder rep we first generate an initial bond configuration according to the disorder value of the first p-T point. Disordered bonds are then added/removed at random to reach the disorder value of the adjacent p-T point. This process is repeated until the bond configuration of the last p-T point is obtained. 
While all disorder reps work on the same set of p-T points, the generation process for these bond configurations are independent and have no correlation between different disorder reps.

For benchmarking purposes, we will start with just one disorder rep and a few p-T points. 


### 2DRBI.cpp 
All meaningful functions regarding the MC simulation are written here.
Most functions take care of the saving and loading of the parameters and observables of each individual core. The most significant function which will be modified over time is ising_sim::update() , which contains all steps executed during a single MC sweep and is called by the main code for every MC step.

### square_rotated.hpp 
Contains the functions used to identify the nearest neighbours of each site.
The Random Bond Ising Model in 2D is used to execute a stabilizer syndrome analysis of the 2D Toric Code with open boundary conditions in the rotated formalism. For a surface code of (odd) lattice size $L\timesL$, the classical model is characterized by a lattice with $L-1$ rows and $(L+1)/2$. Each site $i$ is coupled to $4$ neighbours, with each coupling being the classical equivalent of a quantum qubit. Due to open boundary conditions, the qubits on the top and bottom edges, the leftmost qubit of every second row and the rightmost qubit of every other row have $2$ neighbours, while the qubits on the top left and bottom right corners only have $1$ neighbour.


### FIXED/SUPPLEMENTARY PARTS (code sections that do not require changes)

### main.cpp
Main body of the code. After loading all the parameters the parallel simulations are initialized and executed. Each replica is labeled by its unique temperature value and the respective p value, which are imported from the p-T_points file. Upon conclusion of each simulation, the results are collected, wrapped and stored in an hdf5 file. 

### measurement.cpp, .hpp
Contain the functions used to compute possible observables. At the moment, these are superfluous and are kept only in case of future use. 

### storage_type.hpp
class dictating how data are saved and loaded in hdf5 files.

### temperature.hpp 
Defines the structure for the phase space point associated to each replica. This is solely identified by the temperature and is identified by a lable temp. While in future an extra label p could be added to streamline computation, at the moment it is not required as each replica has its own T value and a unique, respective p value associated to it called from the "p-T_point" table.

### mpi.hpp 
Contains definitions for all objects working on communication between cores.

### pt_adapter.hpp
Contains definitions for all objects using mpi to apply parallel tempering in Monte Carlo simulations.

### exp_beta.hpp
Contains look-up table used to speed up computation of exponentials during Monte Carlo updates.
