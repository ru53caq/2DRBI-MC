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
    L = args.L
    disorder = args.disorder
    timelimit = args.timelimit
    therm=args.therm
    totsweeps=args.totsweeps
    Nreplica=args.Nreplica
    Nbins=args.Nbins
    Seed = args.Seed
    replica_path = args.replica_path

    T_nishimori = 2/ math.log((1-disorder)/disorder)
    T_Top= 1.180

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
    param_dict['__seed'] = '%d' % Seed
    try:
        os.makedirs(replica_path)
    except:
        pass
    sp.call(replace(param_dict) + '< ../../../../%s > %s' % (partemp,parname),cwd=replica_path,shell=True)


    T_vec = np.zeros(Nreplica)
    T_vec[0] = T_nishimori
    for q in range(Nreplica):
        T_vec[q] = T_nishimori + (T_Top - T_nishimori) * q/(Nreplica-1)

    filename = '%s/config_p.data'%(replica_path)
    np.savetxt(filename,config_0,fmt='%1.1d')

    ## Save T-p datapoints in the replica path so simulations know where to look up to
    TPfilename = '%s/T_points.data'%replica_path
    np.savetxt(TPfilename,T_vec,fmt='%1.6f')
if __name__ == "__main__":
    main()
