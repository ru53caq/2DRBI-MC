import math
import h5py
import sys
import numpy as np
import pandas as pd
import dataset
import os
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter
import matplotlib.ticker as mticker
import re
from collections import OrderedDict



base_dir = os.getcwd()

L=5
p="0.100"

h5_name = f"{L}_{p}.h5"
h5_path = os.path.join(base_dir, h5_name)

with h5py.File(h5_path, 'r') as h5_file:
    Z_ts_0dis = h5_file['0dis/results/Z_timeseries'][:]
    dZ_ts_0dis = h5_file['0dis/results/Z_unc_timeseries'][:]
    Z_TOT_0dis = h5_file['0dis/results/Z_TOT']
    dZ_TOT_0dis = h5_file['0dis/results/Z_TOT_unc']
    
    Nreplica = len(h5_file['even/ratioprod'][:])
    for i in range(Nreplica):
        Z_ts_even = h5_file['even/Seed_%i/results/Z_timeseries'%i][:]
        dZ_ts_even = h5_file['even/Seed_%i/results/Z_unc_timeseries'%i][:]
        Z_ts_odd = h5_file['odd/Seed_%i/results/Z_timeseries'%i][:]
        dZ_ts_odd = h5_file['odd/Seed_%i/results/Z_unc_timeseries'%i][:]
        Z_ts_PT = h5_file['PT/Seed_%i/results/Z_timeseries'%i][:]
        dZ_ts_PT = h5_file['PT/Seed_%i/results/Z_unc_timeseries'%i][:]
        Z_TOT_PT = np.float64(h5_file['PT/Seed_%i/results/Z_TOT'%i])
        dZ_TOT_PT = np.float64(h5_file['PT/Seed_%i/results/Z_TOT_unc'%i])
        Z_TOT_even = np.float64(h5_file['even/Seed_%i/results/Z_TOT'%i])
        dZ_TOT_even = np.float64(h5_file['even/Seed_%i/results/Z_TOT_unc'%i])
        Z_TOT_odd = np.float64(h5_file['odd/Seed_%i/results/Z_TOT'%i])
        dZ_TOT_odd = np.float64(h5_file['odd/Seed_%i/results/Z_TOT_unc'%i])
        Z_TOT_MC = Z_TOT_even * Z_TOT_0dis / Z_TOT_odd
        dZ_TOT_MC = np.sqrt( (dZ_TOT_even * Z_TOT_0dis / Z_TOT_odd)**2 + (Z_TOT_even * dZ_TOT_0dis / Z_TOT_odd)**2 + (Z_TOT_even * Z_TOT_0dis* dZ_TOT_odd / Z_TOT_odd**2)**2 ) 

#       Compute the product of the three timeseries
        Z_ts_MC =  Z_ts_even / Z_ts_odd
        Z_ts_MC = Z_TOT_0dis * Z_ts_MC
#        dZ_ts_MC = np.sqrt( (dZ_ts_even * Z_ts_0dis / Z_ts_odd)**2 + (Z_ts_even * dZ_ts_0dis / Z_ts_odd)**2 + (Z_ts_even * Z_ts_0dis* dZ_ts_odd / Z_ts_odd**2)**2 )            
        dZ_ts_MC = np.sqrt( (dZ_ts_even / Z_ts_odd)**2 + (Z_ts_even * dZ_ts_odd / Z_ts_odd**2)**2 )            
        dZ_ts_MC = dZ_ts_MC * Z_TOT_0dis
        plt.errorbar(np.arange(len(Z_ts_MC)), Z_ts_MC, dZ_ts_MC, capsize=5, capthick=1, label="Z_MC_ts")
        plt.errorbar(np.arange(len(Z_ts_MC)), np.ones(len(Z_ts_MC))*Z_TOT_MC, np.zeros(len(Z_ts_MC))*dZ_TOT_MC, capsize=5, capthick=1, label="Z_avg_MC")
        plt.errorbar(np.arange(len(Z_ts_PT)), Z_ts_PT, dZ_ts_PT, capsize=5, capthick=1, label="Z_PT_ts")
        plt.errorbar(np.arange(len(Z_ts_PT)), np.ones(len(Z_ts_PT))*Z_TOT_PT, np.zeros(len(Z_ts_PT))*dZ_TOT_PT, capsize=5, capthick=1, label="Z_avg_PT")






plt.xlabel('t')
plt.ylabel('Z')
plt.grid()
plt.legend()
plt.show()        

