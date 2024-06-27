import numpy as np
import multiprocessing as mp
import subprocess
import json
from timeit import default_timer as timer

def single_run(params):
    """Run a single instance of the Ising model for a given set of parameters."""
    start = timer()
    d = int(params['d'])
    p = float(params['p'])
    beta = float(params['beta'])
    print('beta =', beta)
    bc = str(params['bc'])
    seed = int(params['seed'])
    cmd = ['python', 'thermodynamics.py',
            '--d', f'{d}',
            '--p', f'{p}',
            '--beta', f'{beta}',
            '--bc', f'{bc}',
            '--seed', f'{seed}'
           ]
    if params['overwrite']:
        cmd.append('--overwrite')
    subprocess.run(cmd)
    end = timer()
    print('Run time for d =', params['d'], 'p =', params['p'], 'beta =', params['beta'], 'bc = ', params['bc'], 'seed =', params["seed"], 'is', end - start, 'seconds.')
        
if __name__ == '__main__':
    with open('params.json', 'r') as f:
        params = json.load(f)
    ds = params["ds"]
    ps = np.array(params["ps"])
    betas = np.array(params["betas"])
    bcs = params["bcs"]
    nseeds = params["nseeds"]
    seeds = list(range(nseeds))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        print('cpu count:', mp.cpu_count())
        for d in ds:
            for p, beta in zip(ps, betas):
                for bc in bcs:
                    for seed in seeds:
                        params = {'d': d, 'p': p, 'beta': beta, 'bc': bc, 'seed': seed, 'overwrite': True}
                        pool.apply_async(single_run, args=(params,))
        pool.close()
        pool.join()
        print('All runs complete.')
