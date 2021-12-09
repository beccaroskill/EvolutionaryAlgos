from evolver import Speciman
import csv
import pandas as pd
from vpython import vector, mag, norm, box


n_trials = 10000
i = 3
specs_csv = 'results/ga/n{}_i{}_spec_j10000.csv'.format(n_trials, i)
specs_df = pd.read_csv(specs_csv)
breathe_params = specs_df[['k', 'b', 'c']].values
print(breathe_params)
speciman = Speciman(breathe_params)
start_p, end_p, T = speciman.simulate("shadow", vis=False, save_gif=False, 
                  simulation_time=2.5, plot_energy=False, drop_height=0.02)
print('Speed:  ', mag(end_p - start_p) / T)

dimensions = vector(40, 0.0001, 0.01)
arrow = box(color=vector(1, 0, 0), size=dimensions, axis=norm(end_p - start_p),
                       pos=vector(0.1,0,0.7))
speciman.simulate("shadow", vis=True, save_gif=False, 
                  simulation_time=2.5, plot_energy=False, drop_height=0.02)