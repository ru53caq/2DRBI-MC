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
import argparse
import shutil

def replace(d):
    r = 'sed'
    for i in d:
        r += " -e's:%s:%s:g'" % (i,d[i])
    return r

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, required=True, help='Lattice size')
    parser.add_argument('--disorder', type=float, required=True, help='Disorder parameter')
    parser.add_argument('--timelimit', type=int, required=True, help='Time_limit')
    parser.add_argument('--therm', type=int, required=True, help='Therm_sweeps')
    parser.add_argument('--totsweeps', type=int, required=True, help='Measuring_sweeps')
    parser.add_argument('--Nreplica', type=int, required=True, help='Number_of_reps_per_disorder_config')
    parser.add_argument('--Nbins', type=int, required=True, help='Number_of_histogram_bins')
    parser.add_argument('--Seed', type=int, required=True, help='Random seed')
    parser.add_argument('--init', type=str, choices=['even', 'odd','0dis'], required=True, help='Initialization (even/odd/0dis)')
    parser.add_argument('--replica_path', type=str, required=True, help='path_to_replica_folder')
    args = parser.parse_args()

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
    #    L = 500                            GIVEN
    #    disorder     =  0.100              GIVEN   
    L = args.L
    disorder = args.disorder
    timelimit = args.timelimit
    therm=args.therm
    totsweeps=args.totsweeps
    Nreplica=args.Nreplica
    Nbins=args.Nbins
    Seed = args.Seed
    init = args.init
    replica_path = args.replica_path

    disorder_str =  "%.3f"% disorder
    T_nishimori = 2/ math.log((1-disorder)/disorder)
    T_Top= 1.180
    T_Top_PT=4.5

#    timelimit = 40*3600    GIVEN
    #    therm = 300000         GIVEN
    #    totsweeps = 300000     GIVEN
    #    Nreplica = 10          GIVEN   ##NB: we always use one extra point in p-T space compared to the number of replicas we use
    #    Nbins = 500            GIVEN

    ## Path where each disorder replica will be stored
##    pathformat = 'L_%d/p_%1.3f/%s/Seed_%d'

    ##Initialize the different seeds for each disorder replica
    ##    N_disorder_reps = 600     GIVEN

    ###         THE SLURM CODE TAKES CARE OF THE SEEDS
    #    seed_origin = random.randrange(2**20)
    #   rng = random.Random(seed_origin)
    #    Seeds=np.zeros(N_disorder_reps)
    #    for i in range(N_disorder_reps):
    #        Seeds[i] = random.randint(1,2**20)
    #    print(Seeds)
    #    count = 0

    ##Initialize Lattice features and disorder configuration initializer
    Nx = (L+1)/2    #X-length of the lattice
    Ny = (L-1)      #Y-length of the lattice
    Nboundary_bonds = 2*L   # Number of boundary field terms (NONFLUCTUATING, these are not lattice sites)
    lat_sites = np.intc(Nx*Ny)
    N_bonds = 2*lat_sites - 2*Nx - (Ny-1) + Nboundary_bonds

    config_dis_0=0
    config_0 = np.zeros((lat_sites,2))

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

    config_0, config_dis_0 = config_init(disorder,lat_sites,Nx,Ny)   ##Generates random disorder configuration. 

##    replica_path = pathformat % (L,disorder,init,i)
    full_path = os.getcwd() + '/' + replica_path

    ## Dictate to .ini file
    param_dict['__L'] = '%d' % L
    param_dict['__disorder'] = '%1.3f' % disorder
    param_dict['__timelimit'] = '%d' % timelimit
    param_dict['__therm'] = '%d' % therm
    param_dict['__totsweeps'] = '%d' % totsweeps
    param_dict['__Nreplica'] = '%d' % Nreplica
    param_dict['__Nbins'] = '%d' % Nbins
    param_dict['__init'] = '%s' % init
    param_dict['__seed'] = '%d' % Seed

    try:
        os.makedirs(replica_path)
    except:
        pass
    print(replica_path)

    sp.call(replace(param_dict) + '< ../../../../%s > %s' % (partemp,parname),cwd=replica_path,shell=True)

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
            T_vec[q] = T_nishimori + (T_Top - T_nishimori) * q/(Nreplica)
    if (init == "odd"):
        config[-L:] *= -1
        for q in range(Nreplica+1):
            T_vec[q] = T_nishimori + (T_Top - T_nishimori) * q/(Nreplica)

    if (init == "0dis"):
        config = config*config
        for q in range(Nreplica+1):
            T_vec[q] = T_Top
        p_vec[0]=0.0
        config_dis = 0
        config_dis_0=0
    filename = '%s/config_p=%1.3f.data'%(replica_path,p_vec[0])
    np.savetxt(filename,config,fmt='%1.1d')
    np.savetxt('%s/config_p.data'%replica_path,config,fmt='%1.1d')

    if (init != "0dis"):
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
            filename = '%s/config_p=%1.3f.data'%(replica_path,p_vec[a])
            np.savetxt(filename,config,fmt='%1.1d')
    else:
        flipped_sites = []
        nonflipped_sites = np.arange(lat_sites + L, lat_sites + 2*L).tolist()
        Ntoflip = L
        N_toflip_perstep = np.intc((Ntoflip - Ntoflip%(Nreplica)) / (Nreplica) )
        leftover = Ntoflip%(Nreplica)
        for a in range(1,Nreplica+1):
            for _ in range (N_toflip_perstep):
                fix = random.choice(nonflipped_sites)
                config[np.intc(fix),0] *=-1
                flipped_sites.append(fix)
                nonflipped_sites.remove(fix)
                config_dis += 1/(N_bonds)
            if (leftover>0):
                fix = random.choice(nonflipped_sites)
                config[np.intc(fix),0] *=-1
                flipped_sites.append(fix)
                nonflipped_sites.remove(fix)
                config_dis += 1/(N_bonds)
                leftover += -1

            p_vec[a] = abs(config_dis)
            filename = '%s/config_p=%1.3f.data'%(replica_path,p_vec[a])
            np.savetxt(filename,config,fmt='%1.1d')

    points = np.array([T_vec,p_vec]).transpose()

    ## Save T-p datapoints in the replica path so simulations know where to look up to
    TPfilename = '%s/T-p_points.data'%replica_path
    np.savetxt(TPfilename,points,fmt='%1.6f')

    if (init == "even"):
        T_vec_PT = np.zeros(Nreplica)
        T_vec_PT[0] = T_nishimori
        for q in range(Nreplica):
            T_vec_PT[q] = T_nishimori + (T_Top_PT - T_nishimori) * q/(Nreplica-1)
        ## Save T-p datapoints in the replica path so simulations know where to look up to
        Tfilename = '%s/T_points.data'%replica_path
        np.savetxt(Tfilename,T_vec_PT,fmt='%1.6f')

if __name__ == "__main__":
    main()
