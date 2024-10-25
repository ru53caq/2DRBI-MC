import math
import h5py
import sys
import numpy as np
import pandas as pd
import dataset
import os
from scipy.optimize import curve_fit
import scipy.optimize as optimization
import scipy.stats as stats
from scipy.optimize import leastsq
import scipy.special as sf
import matplotlib
from matplotlib import pyplot as plt
from operator import itemgetter
import matplotlib.ticker as mticker
import re

sys.path.insert(1,'../../../Codes/Python')
np.set_printoptions(threshold=sys.maxsize)



results = {}


base_directory = os.getcwd()  # Replace this with the root directory
output_directory = os.getcwd()   # Directory where output files will be stored

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)


def calculate_ratioprod_and_unc(Zratios, Zratios_unc):
    ratioprod = np.prod(Zratios)
    rel_unc_sq_sum = np.sum((Zratios_unc / Zratios) ** 2)
    ratioprod_unc = ratioprod * np.sqrt(rel_unc_sq_sum)
    return ratioprod, ratioprod_unc



def copy_h5_data(input_h5, output_group):
    """
    Recursively copy all datasets and groups from input_h5 into output_group.
    """
    for key in input_h5:
        item = input_h5[key]
        
        if isinstance(item, h5py.Dataset):  # If it's a dataset, copy the data
            output_group.create_dataset(key, data=item[()])
        
        elif isinstance(item, h5py.Group):  # If it's a group, recursively copy it
            new_group = output_group.require_group(key)
            copy_h5_data(item, new_group)

def process_all_lattices(base_dir, output_dir):
    # Traverse lattice size directories L_*
    for lattice_dir in os.listdir(base_dir):
        lattice_path = os.path.join(base_dir, lattice_dir)
        
        if not os.path.isdir(lattice_path) or not lattice_dir.startswith('L_'):
            continue  # Skip non-directories and irrelevant folders
        
        # Traverse p_* directories
        for p_dir in os.listdir(lattice_path):
            p_path = os.path.join(lattice_path, p_dir)
            
            if not os.path.isdir(p_path) or not p_dir.startswith('p_'):
                continue  # Skip non-directories and irrelevant folders
            
            # Name of the output HDF5 file
            output_file = os.path.join(output_dir, f'{lattice_dir}_{p_dir}.h5')
            
            # Merge the data into this new HDF5 file
            merge_h5_for_lattice_and_p(base_dir, lattice_dir, p_dir, output_file)


def merge_h5_for_lattice_and_p(base_dir, lattice_dir, p_dir, output_file):
    lattice_path = os.path.join(base_dir, lattice_dir)
    p_path = os.path.join(lattice_path, p_dir)
    
    with h5py.File(output_file, 'w') as output_h5:
        # Process both "even" and "odd" directories
        for parity in ['even', 'odd']:
            parity_path = os.path.join(p_path, parity)
            
            if not os.path.isdir(parity_path):
                continue  # Skip if the parity directory does not exist
            
            # Create a group for parity (even or odd) in the output HDF5 file
            parity_group = output_h5.require_group(parity)

            # Prepare vectors for storing ratioprod and ratioprod_unc for each seed
            ratioprod_vec = []
            ratioprod_unc_vec = []


            entries = os.listdir(parity_path)
            # Count only the directories
            Nseeds = sum(1 for entry in entries if os.path.isdir(os.path.join(parity_path, entry)))
            
            # Traverse Seed_* directories under parity
            for i in range(Nseeds):
                seed_path = os.path.join(parity_path, "Seed_%s"%i)
                seed_dir = "Seed_%s"%i
                h5_file_path = os.path.join(seed_path, 'out.h5')                
                if os.path.exists(h5_file_path):
                    with h5py.File(h5_file_path, 'r') as input_h5:
                        Zratios = input_h5['/results/Zratios'][:]
                        Zratios_unc = input_h5['/results/Zratios_unc'][:]

                        ratioprod, ratioprod_unc = calculate_ratioprod_and_unc(Zratios, Zratios_unc)
                        
                        # Append results to vectors
                        ratioprod_vec.append(ratioprod)
                        ratioprod_unc_vec.append(ratioprod_unc)
 
                        seed_group = parity_group.require_group(seed_dir)
                        copy_h5_data(input_h5, seed_group)
                else:
                    print(f"Warning: {h5_file_path} does not exist.")

            # After collecting all ratioprod and ratioprod_unc, store them in the output HDF5 file
            parity_group.create_dataset('ratioprod', data=np.array(ratioprod_vec))
            parity_group.create_dataset('ratioprod_unc', data=np.array(ratioprod_unc_vec))
        
        # Handle the 0dis.h5 file
        dis_file_path = os.path.join(p_path, '0dis/Seed_0/out.h5')
        if os.path.exists(dis_file_path):
            with h5py.File(dis_file_path, 'r') as dis_h5:
                # Create a '0dis' group in the output HDF5 file
                dis_group = output_h5.require_group('0dis')
                
                # Copy all data from 0dis.h5 into the 0dis group
                copy_h5_data(dis_h5, dis_group)
        else:
            print(f"Warning: {dis_file_path} does not exist.")


##even data


process_all_lattices(base_directory, output_directory)



base_dir = os.getcwd()
lattice_sizes = [5,9]
## now we do postprocessing on the Zratios


def Zdivision(z_even, z_odd, unc_even, unc_odd):
    z_final = z_even / z_odd
    z_final_unc = z_final * np.sqrt((unc_even / z_even)**2 + (unc_odd / z_odd)**2)
    return z_final, z_final_unc
def Zmultiplication(z_final, z_tot, z_final_unc, z_tot_unc):
    z_final_tot = z_tot * z_final
    z_final_tot_unc = z_final_tot * np.sqrt((z_final_unc / z_final)**2 + (z_tot_unc / z_tot)**2)
    return z_final_tot, z_final_tot_unc


for lattice_size in lattice_sizes:
    for h5_file_name in os.listdir(os.getcwd()):
        if h5_file_name.endswith('.h5') and h5_file_name.startswith(f'L_{lattice_size}_p_'):
            h5_file_path = os.path.join(os.getcwd(), h5_file_name)
            with h5py.File(h5_file_path, 'r+') as h5_file:
                # Extract Zratios and Zratios_unc from even and odd
                Zratios_even_0 = h5_file['even/ratioprod'][:]
                Zratios_unc_even_0 = h5_file['even/ratioprod_unc'][:]
                Zratios_odd_0 = h5_file['odd/ratioprod'][:]
                Zratios_unc_odd_0 = h5_file['odd/ratioprod_unc'][:]

                # some simulation in even or odd might not have finished: we need to remove those
                min_length = min(len(Zratios_even_0),len(Zratios_odd_0))
                Zratios_even = Zratios_even_0[:min_length]
                Zratios_odd = Zratios_odd_0[:min_length]
                Zratios_unc_even = Zratios_unc_even_0[:min_length]
                Zratios_unc_odd = Zratios_unc_odd_0[:min_length]

                # Extract Z_TOT and Z_TOT_unc from the 0dis group
                Z_TOT = h5_file['0dis/results/Z_TOT'][()]
                Z_TOT_unc = h5_file['0dis/results/Z_TOT_unc'][()]
                # Step 1: Perform the division and propagate uncertainties
                Zratios_final, Zratios_final_unc = Zdivision(
                    Zratios_even, Zratios_odd, Zratios_unc_even, Zratios_unc_odd
                )

                # Step 2: Multiply the result by Z_TOT and propagate uncertainties
                Zratios_final_tot, Zratios_final_tot_unc = Zmultiplication(
                    Zratios_final, Z_TOT, Zratios_final_unc, Z_TOT_unc
                )

                # Store the results back in the same HDF5 file
                h5_file.create_dataset('Ze_o_vec', data=Zratios_final_tot)
                h5_file.create_dataset('Ze_o_unc_vec', data=Zratios_final_tot_unc)

