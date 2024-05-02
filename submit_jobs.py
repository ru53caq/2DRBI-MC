import numpy as np
import os
import subprocess as sp
import random
from scipy.stats import bernoulli as bern



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
    parname = 'RBI.ini'                           ##Name of ini file saved after filling, used for the single replica
    subtemp = 'subfile_template'                    ##Name of sub file template
    subname = 'subfile'                             ##Name of sub file saved after filling, used for the single replica

    ##Define parameters to be dictated to the .ini file
    L = 9
    disorder_max = 0.0
    disorder_min = 0.01

    ##Define set of temperatures and respective disorder values
    #This will define a look-up table to keep tabs on the parameters for when replica exchanges occur
    # For how parallel tempering is implemented, temperatures must go from lowest to highest
    T_vec = [4.,4.5, 5.0, 5.5, 6.0]
    p_vec = [0.,0., 0.0, 0.0, 0.0]
    points = np.array([T_vec,p_vec]).transpose()

    timelimit = 40*3600
    therm = 100000
    totsweeps = 100000
    Nreplica = len(T_vec)-1
    Nbins = 500
    N_avg_p = 1


    ## We try all possible initializations to verify equilibration of disorder averages
    start_config=["even"]

    ## Path where each disorder replica will be stored
    pathformat = 'L_%d/p_%1.3f/%s/Seed_%d'

    ##Initialize the different seeds for each disorder replica
    N_disorder_reps = 1

    seed_origin = random.randrange(2**20)
    rng = random.Random(seed_origin)
    Seeds=np.zeros(N_disorder_reps)
    for i in range(N_disorder_reps):
        Seeds[i] = random.randint(1,2**20)
    print(Seeds)
    count = 0


    for init in start_config:
        for i,Seed in enumerate(Seeds):
            ##Generate path, config file and params.ini file for each iteration

            replica_path = pathformat % (L,disorder_max,init,i)
            full_path = os.getcwd() + '/' + replica_path

            ## Dictate to .ini file
            param_dict['__L'] = '%d' % L
            param_dict['__disorder'] = '%1.3f' % disorder_max
            param_dict['__timelimit'] = '%d' % timelimit
            param_dict['__therm'] = '%d' % therm
            param_dict['__totsweeps'] = '%d' % totsweeps
            param_dict['__Nreplica'] = '%d' % Nreplica
            param_dict['__Nbins'] = '%d' % Nbins
            param_dict['__N_avg_p'] = '%d' % N_avg_p
            param_dict['__init'] = '%s' % init

            param_dict['__seed'] = '%d' % Seed

            ## Dictate to .sub file
            sub_dict['__hour'] = '36'
            sub_dict['__fulldir'] = full_path
            sub_dict['__jobname'] = '2DRBI_%06d' % (Seed)
            try:
                os.makedirs(replica_path)
            except:
                pass

            sp.call(replace(param_dict) + '< ../../../../%s > %s' % (partemp,parname),cwd=replica_path,shell=True)
            sp.call(replace(sub_dict) + '< ../../../../%s > %s' % (subtemp,subname),cwd=replica_path,shell=True)
            #sp.call('cp ../../../../%s .' % exename,cwd=curr_path,shell=True)

            ##Create disorder configuration called by the code
            # Lattice parameters
            Nx = (L+1)/2    #X-length of the lattice
            Ny = (L-1)      #Y-length of the lattice
            Nboundary_bonds = 2*L   # Number of boundary field terms (NONFLUCTUATING, these are not lattice sites)

            lat_sites = np.intc(Nx*Ny)      
            N_bonds = 2*lat_sites - 2*Nx - (Ny-1) + Nboundary_bonds 


            ## Save T-p datapoints in the replica path so simulations know where to look up to
            np.savetxt('%s/T-p_points.data'%replica_path,points,fmt='%1.6f')


            config_dis = 0
            config = np.zeros((lat_sites,2))


            def config_init(p,lat_sites,Nx,Ny):
                ## Disorder bond configuration for the first p-T point
                config = -1 * bern.rvs(p_vec[0], size = (lat_sites + Nboundary_bonds,2)) *2 + 1

                #Adjustment for OPEN BOUNDARY CONDITIONS: no couplings beyond the edges (set certain couplings to 0)
                for i in range(lat_sites + Nboundary_bonds):
                    if (i >= lat_sites - Nx) & (i < lat_sites):
                        config[i,0] = 0         # Last row does not have Jx
                        config[i,1] = 0         # Last row does not have Jy
                    if (i% (2*Nx) == Nx -1) & (i< lat_sites - Nboundary_bonds):
                        config[i,0] = 0         # Right boundaries do not have Jx
                    if (i% (2*Nx) == Nx ) & (i< lat_sites - Nboundary_bonds):
                        config[i,1] = 0         # Left boundaries do not have Jy
                for i in range(lat_sites,lat_sites + Nboundary_bonds):
                    config[i,1] = 0   # only 1 coupling per boundary node

                config_dis = np.count_nonzero(config<0)/(N_bonds)  

                return config,config_dis

            config, config_dis = config_init(p_vec[0],lat_sites,Nx,Ny)
            while(config_dis < p_vec[0]-1e-6):
                config, config_dis = config_init(p,lat_sites,Nx,Ny)

            flipped_sites = np.argwhere(config < 0)
            nonflipped_sites = np.argwhere(config > 0)
            ## Starting from config, remove disordered bonds to get configs for lower disorder values
            # (whenever PT is called, the replica with the new T point will update its p value and load the corresponding disorder config)
            for a,p in enumerate(p_vec):
                if a>0:
                    if (p_vec[a]<p_vec[a-1]):
                        while (config_dis>p+1e-6):
                            fix = random.randint(0, len(flipped_sites)-1)
                            config[flipped_sites[fix][0],flipped_sites[fix][1]] *=-1
                            nonflipped_sites = np.vstack([ nonflipped_sites,flipped_sites[fix] ])
                            flipped_sites = np.delete(flipped_sites,fix, axis=0)
                            config_dis += -1/(N_bonds)

                    if (p_vec[a]>p_vec[a-1]):
                        while (config_dis<p-1e-6):
                            fix = random.randint(0, len(nonflipped_sites)-1)
                            config[nonflipped_sites[fix][0],nonflipped_sites[fix][1]] *=-1
                            flipped_sites = np.vstack([ flipped_sites,nonflipped_sites[fix] ])
                            nonflipped_sites = np.delete(nonflipped_sites,fix, axis=0)
                            config_dis += 1/(N_bonds)
                np.savetxt('%s/config_p=%1.3f.data'%(replica_path,p),config,fmt='%1.1d')


##      UNCOMMENT THESE LINES TO RUN BATCH JOBS ON CLUSTER
#            sp.call('sbatch subfile',cwd=replica_path,shell=True)
#            sp.call('sbatch subfile',cwd=replica_path,shell=True)
        count +=1

if __name__ == "__main__":
    main()
