import numpy as np
import os
import math
import subprocess as sp
import random
from scipy.stats import bernoulli as bern
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter
import matplotlib.ticker as mticker

def replace(d):
    r = 'sed'
    for i in d:
        r += " -e's:%s:%s:g'" % (i,d[i])
    return r

def main():

    ##Template of parameters to be dictated to the .ini file
    param_dict={    '__L':'4',
                    '__disorder':'0.0',
                    '__timelimit':'200000',
                    '__seed':'12345',
                    '__therm':'100000',
                    '__totsweeps':'100000',
                    '__Nreplica':'4',
                    '__Nbins':'1000',
                    '__init':'Hot',
                    '__N_avg_p':'1'
                }

    ##Parameters to be dictated to the .sub file
    sub_dict = {    '__jobname':'Jobtestname',
                    '__fulldir':'./'
                }



    partemp = 'RBI_template.ini'                  ##Name of ini file template
    parname = '2DRBI.ini'                           ##Name of ini file saved after filling, used for the single replica
    subtemp = 'subbatch_template'                    ##Name of sub file template
    subname = 'subbatch'                             ##Name of sub file saved after filling, used for the single replica

    ##Define parameters to be dictated to the .ini file
    L = 5
    disorder     =  0.100                        ##Disorder value used for config generation
    disorder_str = "0.100"
    T_nishimori = 2/ math.log((1-disorder)/disorder)
    T_Top_even= 1.180
    T_Top_odd = 1.1801
    ##Define set of temperatures and respective disorder values
    #This will define a look-up table to keep tabs on the parameters for when replica exchanges occur
    # For how parallel tempering is implemented, temperatures must go from lowest to highest


    timelimit = 40*3600
    therm = 300000
    totsweeps = 300000
    Nreplica = 10   ##NB: we always use one extra point in p-T space compared to the number of nodes we use
    Nbins = 500
    N_avg_p = 1


    ## We try all possible initializations to verify equilibration of disorder averages
    start_config=["even","odd"]

    ## Path where each disorder replica will be stored
    pathformat = 'L_%d/p_%1.3f/%s/Seed_%d'

    ##Initialize the different seeds for each disorder replica
    N_disorder_reps = 600

    seed_origin = random.randrange(2**20)
    rng = random.Random(seed_origin)
    Seeds=np.zeros(N_disorder_reps)
    for i in range(N_disorder_reps):
        Seeds[i] = random.randint(1,2**20)
    print(Seeds)
    count = 0

    ##Initialize Lattice features and disorder configuration initializer

    # Lattice parameters
    Nx = (L+1)/2    #X-length of the lattice
    Ny = (L-1)      #Y-length of the lattice
    Nboundary_bonds = 2*L   # Number of boundary field terms (NONFLUCTUATING, these are not lattice sites)

    lat_sites = np.intc(Nx*Ny)
    N_bonds = 2*lat_sites - 2*Nx - (Ny-1) + Nboundary_bonds

    def config_init(p,lat_sites,Nx,Ny):
        ## Disorder bond configuration for the first p-T point
        configuration = -1 * bern.rvs(p, size = (lat_sites + Nboundary_bonds,2)) *2 + 1
        #Adjustment for OPEN BOUNDARY CONDITIONS: no couplings beyond the edges (set certain couplings to 0)
        for i in range(lat_sites + Nboundary_bonds):
            if (i >= lat_sites - Nx) & (i < lat_sites):
                configuration[i,0] = 0         # Last row does not have Jx
                configuration[i,1] = 0         # Last row does not have Jy
            if ( (i% (2*Nx) ) == Nx -1) & (i < lat_sites):
                configuration[i,0] = 0         # Right boundaries do not have Jx
            if (i% (2*Nx) == Nx ) & (i< lat_sites):
                configuration[i,1] = 0         # Left boundaries do not have Jy
        for i in range(lat_sites,lat_sites + Nboundary_bonds):
            configuration[i,1] = 0   # only 1 coupling per boundary node

        configuration_disorder = np.count_nonzero(configuration<0)/(N_bonds)

        return configuration,configuration_disorder

    for i,Seed in enumerate(Seeds):

        config_dis_0 = 0
        config_0 = np.zeros((lat_sites,2))

        config_0, config_dis_0 = config_init(disorder,lat_sites,Nx,Ny)   ##Generates random disorder configuration. 

        for init in start_config:
            ##Generate path, config file and params.ini file for each iteration

            replica_path = pathformat % (L,disorder,init,i)
            full_path = os.getcwd() + '/' + replica_path

            ## Dictate to .ini file
            param_dict['__L'] = '%d' % L
            param_dict['__disorder'] = '%1.3f' % disorder
            param_dict['__timelimit'] = '%d' % timelimit
            param_dict['__therm'] = '%d' % therm
            param_dict['__totsweeps'] = '%d' % totsweeps
            param_dict['__Nreplica'] = '%d' % Nreplica
            param_dict['__Nbins'] = '%d' % Nbins
            param_dict['__N_avg_p'] = '%d' % N_avg_p
            param_dict['__init'] = '%s' % init
            param_dict['__seed'] = '%d' % Seed
            
            try:
                os.makedirs(replica_path)
            except:
                pass

            sp.call(replace(param_dict) + '< ../../../../%s > %s' % (partemp,parname),cwd=replica_path,shell=True)


            #sp.call('cp ../../../../%s .' % exename,cwd=curr_path,shell=True)


            ## Starting from config, remove disordered bonds to get configs for lower disorder values
            # (whenever PT is called, the replica with the new T point will update its p value and load the corresponding disorder config)

            ## Every step I have to flip N_toflip_perstep spins
            ## Among these, I have to sprinkle the leftovers: I could just put them at the first "leftover" steps 
            ## p will slowly decrease with constant jumps (a bit larger for the first "leftover" steps) until it reaches 0

            config = np.copy(config_0)
            config_dis = config_dis_0
            flipped_sites = np.argwhere(config_0 < 0)
            nonflipped_sites = np.argwhere(config_0 > 0)

            Nflipped = len(flipped_sites)
            N_toflip_perstep = np.intc((Nflipped - Nflipped%(Nreplica)) / (Nreplica) )
            leftover = Nflipped%(Nreplica)


            T_vec = np.ones(Nreplica+1) *T_nishimori
            p_vec = np.zeros(Nreplica+1)

            p_vec[0] = config_dis_0
            T_vec[0] = T_nishimori             ## First point always starts on the Nishimori line point given by p

            if (init == "even"):
                for q in range(Nreplica+1):
                    T_vec[q] = T_nishimori + (T_Top_even - T_nishimori) * q/(Nreplica)
            if (init == "odd"):
                config[-L:] *= -1
                for q in range(Nreplica+1):
                    T_vec[q] = T_nishimori + (T_Top_odd - T_nishimori) * q/(Nreplica)

            np.savetxt('%s/config_p=%1.3f.data'%(replica_path,p_vec[0]),config,fmt='%1.1d')

            for a in range(1,Nreplica+1):
                for _ in range (N_toflip_perstep):
                    fix = random.randint(0, len(flipped_sites)-1)
                    config[flipped_sites[fix][0],flipped_sites[fix][1]] *=-1
                    nonflipped_sites = np.vstack([ nonflipped_sites,flipped_sites[fix] ])
                    flipped_sites = np.delete(flipped_sites,fix, axis=0)
                    config_dis += -1/(N_bonds)
                if (leftover>0):
                    fix = random.randint(0, len(flipped_sites)-1)
                    config[flipped_sites[fix][0],flipped_sites[fix][1]] *=-1
                    nonflipped_sites = np.vstack([ nonflipped_sites,flipped_sites[fix] ])
                    flipped_sites = np.delete(flipped_sites,fix, axis=0)
                    config_dis += -1/(N_bonds)
                    leftover += -1

                p_vec[a] = abs(config_dis)
                np.savetxt('%s/config_p=%1.3f.data'%(replica_path,p_vec[a]),config,fmt='%1.1d')


            points = np.array([T_vec,p_vec]).transpose()

            ## Save T-p datapoints in the replica path so simulations know where to look up to
            np.savetxt('%s/T-p_points.data'%replica_path,points,fmt='%1.6f')

##      UNCOMMENT THESE LINES TO RUN BATCH JOBS ON CLUSTER
#            sp.call('sbatch subfile',cwd=replica_path,shell=True)
        count +=1

    ##Fix a number of jobs that should be executed for each slurm call proportional to N_reps
    N_jobs_per_slurm = 100

    N_slurms = np.intc(N_disorder_reps/N_jobs_per_slurm)

    for i in range(N_slurms):
        
        ## Dictate to .sub file
        sub_dict['__L'] = L
        sub_dict['__p'] = disorder_str
        sub_dict['__hour'] = '36'
        sub_dict['__fulldir'] = full_path
        sub_dict['__jobname'] = '2DRBI'
        sub_dict['__seed_i'] = N_jobs_per_slurm*i
        sub_dict['__seed_f'] = N_jobs_per_slurm*(i+1)
        sp.call(replace(sub_dict) + '<%s> %s' % (subtemp,subname),cwd="./",shell=True)
        sp.call('sbatch subbatch',cwd="./",shell=True)

if __name__ == "__main__":
    main()
