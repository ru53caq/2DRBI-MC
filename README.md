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

## Execution of the code

Numerical implementation of the MC simulation requires the generation of a set of initial configurations and simulation parameters. This process is carried out by [`submit_jobs.py`](./submit_jobs.py), which follows the footprint given by [`RBI_template.ini`](./RBI_template.ini).

Once the configuration and the parameter files are generated, one can execute the command 
```bash
$ mpirun -n 1 ~/path_to_build/2DRBI ../path_to_ini_file/2DRBI.ini
```
to run the code with the chosen parameters and generated configurations.

In its current version, the code will give as output the amount of time it spent in each p-T_point of choice. 
The code is embarassingly parallelized, meaning it will run the same computation on different cores without communication. 

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
