from symbolic_regression import VisualizeSearch, load_dataset

if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 100000
    

    fitness_f = lambda x : 100 - x
    VisualizeSearch.plot_fitness('results/rmhc_100restarts_ops6to8/', 'n{}'.format(n_trials), 
                                'rmhc (100 restarts)', 'RMHC Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    VisualizeSearch.plot_fitness('results/random_ops6to8/', 'n{}'.format(n_trials), 
                                'random', 'Random Search', 
                                fitness_f=fitness_f, ylim=(0,100))