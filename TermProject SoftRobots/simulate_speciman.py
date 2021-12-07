from evolver import Speciman
import csv
import pandas as pd

n_trials = 10000
i = 5
specs_csv = 'results/ga/n{}_i{}_spec_j6439.csv'.format(n_trials, i)
specs_df = pd.read_csv(specs_csv)
breathe_params = specs_df[['k', 'b', 'c']].values
print(breathe_params)
speciman = Speciman(breathe_params)
speciman.simulate("shadow", vis=True, save_gif=False, 
                  simulation_time=2, plot_energy=False, drop_height=0.02)