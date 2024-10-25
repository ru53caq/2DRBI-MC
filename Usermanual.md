# 2DRBI-MC
MC code for computing the free energy ratio of the 2DRBI model for varying points in p-T space.

2 methods are implemented

1. Daniel Loss (2DRBI_PT)
	- Simulation of N replicas at N temperature points with a fixed disorder configuration
	- Line updates between even and odd boundary conditions are proposed at all T, but only accepted in disordered phase
	- Exchanges of configurations between replicas at different temperatures allows for percolation of even/odd boundary configurations at low T
	- Zeven/Zodd computed as ratio of the amount of time the replica at T_nishimori is in the even or odd boundary configuration
	
2. Partitioned Zratio calculation (2DRBI_MC)
	- calculate Z_even/Z_odd by passing through the 0-disorder axis
	- Simulation is divided in three sections:
		a.	simulation	of Z_even, where N replicas are generated starting from a maximally disordered configuration; the other configurations are obtained by removing 
		disorder bonds until no error is present. For each replica, run a MC simulation proposing updates between its configuration and the adjacent one (dE~cost). 
		These should be easily viable if enough replicas are initialized. At the end of the simulation, take the product of Zratios and expand uncertainties to obtain Z_TOT, dZ_TOT 
		between the disordered and the 0-disorder configuration with even boundary conditions. We will call this output Z_even
		b.     Do the same, this time with all bonds across one line flipped for each configuration as to introduce the odd boundary conditions. 
		This simulation, after taking product of Zratios, will give Z_TOT and dZ_TOT  between the disordered and the 0-disorder configuration with odd boundary conditions. 
		We will call this output Z_odd
		c.	Run another simulation connecting the two configurations with even and odd BC with zero disorder, always using several replicas to ensure the probability 
		of an exchange between configurations being accepted is dE~cost . THIS NEEDS TO BE DONE ONLY ONCE PER SYSTEM SIZE FOR A FIXED TEMPERATURE T_0dis. 
		We will call this output Z_0dis.
		d.	Compute Z = Z_even* Z_0dis / Z_odd

	
3. Simulation
	Simulations are run by submitting subjobs_MC.sub.
	The code requires:
	L : Linear system size
	p : disorder value
	therm : number of thermalization sweeps
	totsweeps : number of measurement sweeps to take after thermalization
	Nreplica : number of replicas simulated in the chains
	T_Top : maximum temperature used for the 2DRBI_MC method (i.e. the value the replicas at 0 disorder are simulated)
	T_Top_PT : maximum temperature used for the 2DRBI_PT method (i.e. maximum temperature for the PT chain to reach the disordered phase)
	PT : Flag to decide whether to also run the simulation using the PT method
	N_disorder_reps : Number of disorder configurations that will be simulated IN SERIES
	
	The workflow is the following:
	 
	 1. Generate $N_disorder_reps seeds
	 2. For each seed, generate the relevant disorder configurations for even BC; the same are generated for odd BC + a Z line change. 
	 These are stored in L_$L/p_$p/$init/Seed_$Seed     where $L,$p,$Seed are specific for the replica, and $init can vary be either "even","odd"
	 3. Run the simulation for even BC. Outcomes are stored in the L_$L/p_$p/even/Seed_--Seed/out.h5  for a given replica
	 4. Run the simulation for odd BC. Outcomes are stored in the L_$L/p_$p/odd/Seed_--Seed/out.h5  for a given replica
	 5. If PT = "True, Run the simulation using 2DRBI_PT. Outcomes are stored in the L_$L/p_$p/PT/Seed_$Seed/out.h5  for a given replica
	 6. Once all replicas are done, run the 0disorder simulation. a mul factor can be tuned to run the simulation for longer in order to increase accuracy of the outcomes. 
	 (Since this is run only once, the uncertainty of these outcomes should be negligible) . Outcomes are stored in the L_$L/p_$p/0dis/Seed_$Seed/out.h5 
	 
4. Tuning simulation parameters:
	The number of replicas and the distribution of points in P-T space is crucial for the efficiency of the algorithms.
	As a rule of thumb, we will deal with 2*(L+1)*(L-1) + 2*L bonds at p ~ 0.1. On average, setting #nodes/replicas between ~ [L,2L] should be enough for the MC code
	 to propose updates at a good rate.
	There is also dependance on T_Top and T_Top_PT , as acceptance rates between configurations is heavily affected by the temperature difference of replicas. 
	For the MC method, setting T_Top ~ 1.1 (Replicas are generated at temperatures between T_nishimori and T_top: if this line does not cross the phase transition it should be fine.
	 When in doubt, T_top = T_Nishimori can also be used, but autocorrelation times might increase. In general, T_Top must be below 1.5).
	 
	 For T_Top_PT the situation is more complicated, as acceptance rates are solely determined by the separation in temperature between adjacent replicas.
	 In general, checking whether exchanges are accepted is enough to say the system will reach equilibration. An acceptance rate of ~ 0.3-0.4 for each replica is good enough.
	 T_Top_PT must be in the disordered phase in order to ensure the acceptance of line-flips to go from even to odd BC. 
	 At p=0 in the thermodynamic limit, this is T=2.1, so T_Top_PT should be at least 2.5.

	2 sanity checks are useful: 
	
		a. first, ensure that T_Top_PT is disordered, i.e. line updates are accepted. This can be done by adding a flag around line 135 of 2DRBI_PT.cpp (like:  if (temp.temp == T_vec[T_vec.end()]) { std::cout << " ACC  at sweep   " << sweeps << std::endl; }  
	
		b. Ensure that updates between replicas are accepted with decent frequency (acceptance rates should ideally be constant and around 0.5).
		As the system size increases, swaps become more rare and the number of replicas should grow significantly: one should set a large enough number of replicas to span the whole phase space. While for L=5,7 5-10 replicas can work, for larger system sizes this should be set to 20 or 30 (to be safe, more is better but slower). 
		
		To get a feeling, comment out lines 65,113 of subjobs_MC (so only PT sims are executed) and play around at L=7, L=9 with T_Top_PT ~ [2,2.5,3,3.5] (or just generate configurations and run them on local computer)

		NB: If one wants to define new global variables to play around with acceptance rates, one first needs to initialize them in the 2DRBI.hpp file. 
		In line 153 of 2DRBI_PT.cpp, a conditional "pt_checker" shows whether an exchange for a replica at temperature temp.temp was updated or not and what is the new temperature.
		( For testing purposes, temp.temp gives the current temperature of the replica, temp_old.temp gives the temperature of the old replica ). 
	example:
        if (temp.temp == T_vec[5]){
            n_acc +=1;
        }
        if (sweeps > thermalization_sweeps+total_sweeps - 1000)
            std::cout << n_acc / (sweeps/pt_sweeps) << std::endl;


		[If one wants to run both MC and PT in the same submission, the code would use the same number of tasks for both: this might not be "fair", as MC needs less nodes than PT, 
		and should be updated. In the meantime, setting an arbitrarily large number of nodes does no harm to MC (aside from slightly increasing uncertainties), so one should 
		prioritize the needs of the PT simulations ]

	Good testing values for L up to 9  (large number of sweeps and replicas, but should give nice outputs): 
	Ncores = 18
	therm = 3000000		
	totsweeps = 3000000
	T_Top = 1.18
	T_Top_PT = 2.5		(might want to set this to 2.0 and 2.5)