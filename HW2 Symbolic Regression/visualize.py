from symbolic_regression import VisualizeSearch, load_dataset
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 10000


    fitness_f = lambda x : (100 - x)
    VisualizeSearch.plot_fitness('results/ga_diverse/', 'n{}'.format(n_trials), 
                                'GA \n(population=1000,\nselection=30%)', 'GA (Diverse) Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    VisualizeSearch.plot_fitness('results/ga/', 'n{}'.format(n_trials), 
                                'GA \n(population=100,\nselection=20%)', 'GA Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    VisualizeSearch.plot_fitness('results/rmhc_100restarts_depth3to8/', 'n{}'.format(n_trials), 
                                'RMHC (100 restarts)', 'RMHC Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    VisualizeSearch.plot_fitness('results/rmhc_depth3to8/', 'n{}'.format(n_trials), 
                                'RMHC', 'RMHC Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    VisualizeSearch.plot_fitness('results/random_depth3to8/', 'n{}'.format(n_trials), 
                                'Random', 'Random Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    plt.title('Learning curves across algorithms')
    plt.savefig('figs/all_algo_fitness.png')
    plt.ylim(98, 100)
    plt.xlim(10**3, 10**5)
    plt.savefig('figs/all_algo_fitness_zoomed.png')
    plt.show()
