# 2DRBI-MC

Monte Carlo (MC) code for computing the free energy ratio of the 2DRBI model across various points in $p - T$ space.

---

## Overview

The 2DRBI-MC code provides two simulation methods for computing the ratio $Z_{\text{even}} / Z_{\text{odd}}$:  
1. **Partitioned Z-ratio Calculation (2DRBI_MC)**  
2. **Parallel Tempering (PT) Method (2DRBI_PT)**  

The methods differ in their approach, detailed in **Simulation Methods**.

---

## Building and Installation

### Requirements

To run the code, the following are required:
- **ALPSCore libraries** (instructions available on ALPSCore's wiki)
- **CMake**
- **Eigen3 library**

### Build Steps

```bash
$cmake .. -DCONFIG_MAPPING=LAZY -DALPSCore_DIR=~/path_to_ALPSCore -DEigen3_DIR=~path_to_eigen3/eigen3
$make -j8
```


## Simulation Methods

### 1. Partitioned Z-ratio Calculation (2DRBI_MC)

This method calculates $Z_{\text{even}} / Z_{\text{odd}}$ by passing through the zero-disorder axis. The process is divided into the following steps:

1. **Simulation of $Z_{\text{even}}$:**  
   - $N$ replicas are generated starting from an initial disordered configuration.
   - Adjacent configurations are obtained by removing disorder bonds until no disorder is present.
   - For each replica, a Monte Carlo simulation is run proposing updates between configurations. Exchanges between replicas are accepted with a probability proportional to $\exp(-\Delta E)$.
   - At the end, compute the product of intermediate $Z$-ratios to obtain $Z_{\text{even}}$ and its uncertainty $dZ_{\text{even}}$.

2. **Simulation of$Z_{\text{odd}}$:**  
   - Similar to $Z_{\text{even}}$, but with one line of flipped bonds to introduce odd boundary conditions.
   - Compute $Z_{\text{odd}}$ and $d Z_{\text{odd}}$ using the same procedure.

3. **Zero Disorder Simulation ($Z_{0,\text{dis}}$):**  
   - A simulation connects the configurations with even and odd boundary conditions at zero disorder.
   - This step is performed only once per system size for a fixed temperature $T_{0,\text{dis}}$.
   - Compute $Z_{0,\text{dis}}$ and $d Z_{0,\text{dis}}$.

4. **Final Ratio Computation:**  
   Compute the free energy ratio:  
   \[
   Z = \frac{Z_{\text{even}} \cdot Z_{0,\text{dis}}}{Z_{\text{odd}}}
   \]

This method is embarassingly parallelized


---

### 2. Parallel Tempering (2DRBI_PT)

This method involves running $N$ replicas at $N$ temperature points with the same fixed disorder configuration. Key steps include:

1. **Line Updates:**  
   - Propose line updates between even and odd boundary conditions at the highest temperature ($T \to \infty$).

2. **Replica Exchanges:**  
   - Swap configurations between replicas at adjacent temperatures to ensure efficient sampling.
   - Low-temperature configurations can percolate between even and odd boundaries.

3. **Ratio Computation:**  
   Compute $Z_{\text{even}} / Z_{\text{odd}}$ as the ratio of time spent in even and odd configurations at $T_{\text{Nishimori}}$.

---

## Running the Code

Simulations are run by submitting the `subjobs_MC.sub` script. Key parameters to set:

- `L`: Linear system size
- `p`: Disorder value
- `therm`: Number of thermalization sweeps
- `totsweeps`: Number of post-thermalization sweeps
- `Nreplica`: Number of replicas simulated
- `T_Top`: Maximum temperature for the 2DRBI_MC method
- `T_Top_PT`: Maximum temperature for the 2DRBI_PT method
- `PT`: Flag to enable/disable the PT method
- `N_disorder_reps`: Number of disorder configurations simulated sequentially

### Workflow

1. Generate `N_disorder_reps` seeds.
2. For each seed, create the relevant disorder configurations for even and odd boundary conditions using `genconfigs_MC.py`.  
   These are stored in directories:  
   `L_$L/p_$p/$init/Seed_$Seed`, where `$init` can be `even` or `odd`.
3. Run simulations for even and odd BC. Results are saved to:  
   `L_$L/p_$p/even/Seed_$Seed/out.h5`  
   `L_$L/p_$p/odd/Seed_$Seed/out.h5`
4. If `PT=True`, run a simulation for the PT method. Results are stored in:  
   `L_$L/p_$p/PT/Seed_$Seed/out.h5`
5. Once all replicas are completed, run the zero-disorder simulation. Adjust the `mul` factor to improve accuracy. Results are stored in:  
   `L_$L/p_$p/0dis/Seed_$Seed/out.h5`

All results of the simulations are then collected into a single file by [`datacollect.py`] and stored in (./datacollect.py)

---

## Tuning Simulation Parameters

The efficiency of the simulations depends on careful tuning of parameters:

- **Number of Replicas:**  
  For $p \sim 0.1$, set the number of replicas between $L$ and $2L$.

- **Temperature Distribution:**  
  - For the MC method,$T_{\text{Top}}$should be slightly above$T_{\text{Nishimori}}$, typically$T_{\text{Top}} \approx 1.1$. Ensure it doesn’t cross the phase transition.  
  - For the PT method,$T_{\text{Top\_PT}}$must be in the disordered phase. A good starting point is$T_{\text{Top\_PT}} \geq 2.5$.

- **Sanity Checks:**  
  1. Ensure line updates at $T_{\text{Top\_PT}}$are accepted frequently.
  2. Monitor acceptance rates for replica exchanges (target $\sim 0.3-0.4$).

Example parameters for $L = 9$,$p = 0.1$:  
```bash
Ncores = 18
therm = 300000
totsweeps = 1000000
T_Top = 1.18
T_Top_PT = 1000



## Code sections
### SIGNIFICANT PARTS (sections which require changes depending on parameter choice, code updates etc):

### submit_jobs.py
Generates a sequence of simulations for a given choice of parameters. Here, one needs to choose:

$Lattice size                          L
$a list with N temperature values	    T_vec
$a list with N disorder values		    p_vec
$Number of thermalization sweeps		therm
$Number of simulation sweeps		    totsweeps
$Number of different configurations	N_disorder_reps
$start configuration (even or odd)	    start_config

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
