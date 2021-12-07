from evolver import Search
import csv
import pandas as pd

search_mgr = Search()
n_trials = 10000
for i in range(5, 6):
    df, best_specimen = search_mgr.run_ga_parallel(n_trials, num_nodes=1, n_pop=2, 
                                                   top_k=1, trial_name=i, sim_time=4)
    results_subdir = 'results/ga'
    df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
    df_spec = pd.DataFrame(best_specimen[-1].breathe_params, columns=['k', 'b', 'c'])
    df_spec.to_csv('{}/n{}_i{}_spec.csv'.format(results_subdir, n_trials, i))
    # plt.figure(figsize=(6, 6))
    # VisualizeSearch.plot_f(best_specimen[-1], dataset)
    # plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
    # plt.show() 