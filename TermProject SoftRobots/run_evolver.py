from evolver import Search
import csv
import pandas as pd

search_mgr = Search()
n_trials = 10000
for i in range(4, 5):
    df, best_specimen = search_mgr.run_ga_parallel(n_trials, num_nodes=None, n_pop=100, 
                                                   top_k=30, trial_name=i, sim_time=0.5)
    results_subdir = 'results/ga'
    df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
    for j in range(30, len(best_specimen)):
        if best_specimen[j] and best_specimen[j] != best_specimen[j-1]:
            df_spec = pd.DataFrame(best_specimen[j].breathe_params, columns=['k', 'b', 'c'])
            df_spec.to_csv('{}/n{}_i{}_spec_j{}.csv'.format(results_subdir, n_trials, i, j))
    # plt.figure(figsize=(6, 6))
    # VisualizeSearch.plot_f(best_specimen[-1], dataset)
    # plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
    # plt.show() 
