"""
Rebecca Roskill

rcr2144

MECS 4510

Prof. Hod Lipson

Submitted Sunday, September 19th, 2021

Grace hours used: 0

Grace hours remaining: 96
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

plt.figure(figsize=(7, 7))

results_dir = 'results/'
experiment_name = 'TSP 1: Shortest Path (varying mutation-recombination ratio)'
                  
trial_batches = {'random': [],
                 'rmhc': [],
                 'ga_': [],
                 'gal_': [],
                 'gal2': []}

trial_batches = {'ga_n100000_m0.2_unordered': [],
                'ga_n100000_m0.5_unordered': [],
                'ga_n100000_m0.8_unordered': [],
                'ga_n100000_m0.9_unordered': []}

# batch_names = {'random': "Random",
#                  'rmhc': "RMHC",
#                  'ga_': "GA (No linkage)",
#                  'gal_': "GA (Reverse linkage -> tight linkage)", #['swap cross-region', 'hybrid', 'swap intra-region'], recombination_types=['cross regions', 'hybrid', 'intra-region']
#                  'gal2': "GA (Reverse linkage -> hybrid linkage)"} #['swap cross-region', 'hybrid'], recombination_types=['cross regions', 'hybrid']


batch_names = { 'random_n100000': 'Random',
                'rmhc_n100000': 'RMHC',
                'ga_kmeans_hybrid_n10000': 'GA k-Means Hybrid',
                'rmhc_kmeans_hybrid_n10000': 'RMHC k-Means Hybrid',
                'ga_n100000_m0.2_i': 'GA mutation=0.2, recombination=0.8\npop=100 (linked by region)',
                'ga_n100000_m0.2_unordered': 'GA mutation=0.2, recombination=0.8\npop=100 (no linkage)',
                'ga_n100000_m0.5_i': 'GA mutation=0.5, recombination=0.5\npop=100 (linked by region)',
                'ga_n100000_m0.5_unordered': 'GA mutation=0.5, recombination=0.5\npop=100 (no linkage)',
                'ga_n100000_m0.8_i': 'GA mutation=0.8, recombination=0.2\npop=100 (linked by region)',
                'ga_n100000_m0.8_unordered': 'GA mutation=0.8, recombination=0.2\npop=100 (no linkage)',
                'ga_n100000_b100_m0.8_p0.2_unordered': 'GA mutation=0.8, recombination=0.2\npop=100, p=0.2 (no linkage)',
                'ga_n100000_m0.9_i': 'GA mutation=0.9, recombination=0.1\npop=100 (linked by region)',
                'ga_n100000_m0.9_unordered': 'GA mutation=0.9, recombination=0.1\npop=100 (no linkage)',
                'ga_n100000_b1000_m0.5_ordered': 'GA mutation=0.5, recombination=0.5\npop=1000 (linked by region)',
                'ga_n100000_b1000_m0.5_unordered': 'GA mutation=0.5, recombination=0.5\npop=1000 (no linkage)',
                'ga_n100000_b1000_m0.8_ordered': 'GA mutation=0.8, recombination=0.2\npop=1000 (linked by region)',
                'ga_n100000_b1000_m0.8_unordered': 'GA mutation=0.8, recombination=0.2\npop=1000 (no linkage)',
                'ga_n100000_b50_m0.5_ordered': 'GA mutation=0.5, recombination=0.5\npop=50, p=0.2 (linked by region)',
                'ga_n100000_b50_m0.5_unordered': 'GA mutation=0.5, recombination=0.5\npop=50, p=0.2 (no linkage)',
                'ga_n100000_b50_m0.8_ordered': 'GA mutation=0.8, recombination=0.2\npop=50, p=0.2 (linked by region)',
                'ga_n100000_b50_m0.8_unordered': 'GA mutation=0.8, recombination=0.2\npop=50, p=0.2 (no linkage)'}
    
for f in os.listdir(results_dir):
    if 'csv' in f:
        df = pd.read_csv(results_dir + f)
        
        for trial_batch in trial_batches:
            if trial_batch in f:
                trial_batches[trial_batch] += [df]
                print(trial_batch)

for trial_batch in trial_batches:
    dfs = trial_batches[trial_batch]
    trial_batch = batch_names[trial_batch]

    trials = dfs[0]['trial']
    best_dist = -np.mean([df['best_distance'] for df in dfs], axis=0)
    std_err_pts = np.logspace(4, np.log10(max(trials)), 10).astype(int)
    best_dist_err = [np.std([-df['best_distance'][i] for df in dfs]) 
                     for i in std_err_pts]
    plt.subplot(111)
    plt.plot(trials, best_dist, label=trial_batch)
    best_dist_c = plt.gca().lines[-1].get_color()
    plt.errorbar(std_err_pts, [best_dist[i] for i in std_err_pts], 
                  yerr=best_dist_err, linestyle='', capsize=2, c=best_dist_c)
    plt.title(experiment_name)
    plt.xlabel('Trial')
    plt.xscale('log')
    plt.xlim((10000, 100000))
    plt.ylabel('Fitness')
    plt.ylim((-400, -150))
    plt.legend(loc='lower right')

plt.savefig('figs/{}.png'.format('mr_ratio_zoomed.png'), dpi=300)
