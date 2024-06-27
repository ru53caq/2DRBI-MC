import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
plt.style.use("figures/norm.mplstyle")


if __name__ == '__main__':
    with open('params.json', 'r') as f:
        params = json.load(f)
    ds = params["ds"]
    ps = np.array(params["ps"])
    betas = np.array(params["betas"])
    bcs = params["bcs"]
    nseeds = params["nseeds"]
    seeds = list(range(nseeds))
    data_directory = "data"

    data_tensor = np.zeros((len(ds), len(ps), len(bcs), len(seeds), 4))  # d, p, bc, seed, [pf, energy, mag, free_energy]

    
    for id, d in enumerate(ds):
        for ip, (p, beta) in enumerate(zip(ps, betas)):
            print(f'd={d}, p={p}, beta={beta}')
            for ibc, bc in enumerate(bcs):
                for seed in seeds:
                    filename = f'd={d}_p={p}_beta={beta}_bc={bc}_seed={seed}.pkl'
                    data = pd.read_pickle(f'{data_directory}/{filename}')
                    data_tensor[id, ip, ibc, seed, :] = [data['partition_function'], data['energy'], data['magnetization'], data['free_energy']]
                    
    np.save('data_tensor.npy', data_tensor)

    # take average over seeds
    partition_function_mean = np.mean(data_tensor[:, :, :, :, 0], axis=3)
    energy_mean = np.mean(data_tensor[:, :, :, :, 1], axis=3)
    magnetization_mean = np.mean(data_tensor[:, :, :, :, 2], axis=3)
    magnetization_std = np.std(data_tensor[:, :, :, :, 2], axis=3)
    free_energy_mean = np.mean(data_tensor[:, :, :, :, 3], axis=3)

    assert free_energy_mean.shape == (len(ds), len(ps), len(bcs))

    with open('partition_function_temperature_ratios.txt', 'w') as f:
        partition_function_even = partition_function_mean[0, :, 0]
        ratios = partition_function_even[0:-1]/partition_function_even[1:]
        f.write('Partition function ratio\n')
        for i, p in enumerate(ps[:-1]):
            f.write(f'Z({betas[i]})/Z({betas[i+1]})={ratios[i]}\n')

    with open('partition_function_even_odd_ratios.txt', 'w') as f:
        partition_function_even = partition_function_mean[0, :, 0]
        partition_function_odd = partition_function_mean[0, :, 1]
        ratios = partition_function_even/partition_function_odd
        f.write('p\tPartition function ratio\n')
        for i, p in enumerate(ps):
            f.write(f'temperature={1/betas[i]}, p={p}, even/odd ratio={ratios[i]}\n')
        

    # # plot free energy
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for i, bc in enumerate(bcs):
    #     for id, d in enumerate(ds):
    #         ax[i].plot(ps, free_energy_mean[id, :, i]/((d-1)*(d//2+1)), label=f'bc={bc}, d={d}')
    #         ax[i].set_xlabel('p')
    #         ax[i].set_ylabel('Free energy')
    #         ax[i].legend()
    #         ax[i].set_title('Free energy vs p')

    # plt.savefig('figures/free_energy.pdf', bbox_inches='tight')

    # plot magnetization
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for i, bc in enumerate(bcs):
    #     for id, d in enumerate(ds):
    #         ax[i].plot(ps, magnetization_mean[id, :, i]/((d-1)*(d//2+1)), label=f'bc={bc}, d={d}')
    #         ax[i].errorbar(ps, magnetization_mean[id, :, i]/((d-1)*(d//2+1)), yerr=magnetization_std[id, :, i]/((d-1)*(d//2+1)), fmt='o')
    #         ax[i].set_xlabel('p')
    #         ax[i].set_ylabel('Magnetization')
    #         ax[i].set_ylim(0, 1)
    #         ax[i].legend()
    #         ax[i].set_title('Magnetization vs p')
        
    # plt.savefig('figures/magnetization.pdf', bbox_inches='tight')
    
    # plot entropy
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # for i, bc in enumerate(bcs):
    #     for id, d in enumerate(ds):
    #         entropy = (energy_mean[id, :, i] - free_energy_mean[id, :, i]) * betas / ((d-1)*(d//2+1))
    #         ax[i].plot(ps, entropy, label=f'bc={bc}, d={d}')
    #         ax[i].set_xlabel('p')
    #         ax[i].set_ylabel('Entropy')
    #         ax[i].legend()
    #         ax[i].set_title('Entropy vs p')

    # plt.savefig('figures/entropy_free_energy.pdf', bbox_inches='tight')
    # plt.show()


