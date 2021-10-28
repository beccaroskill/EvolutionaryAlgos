from symbolic_regression import VisualizeSearch, load_dataset
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 10000


    fitness_f = lambda x : (100 - x)
    for i in range(4, 9):
        VisualizeSearch.plot_fitness('results/ga_complexity/', 'depth{}'.format(i), 
                                'GA \n(Max depth={})'.format(i), 'GA (Diverse) Search', 
                                fitness_f=fitness_f, ylim=(0,100))
    plt.title('Learning curves across GA complexity')
    plt.savefig('figs/ga_complexity_fitness.png')
    plt.ylim(95, 100)
    plt.xlim(10**3, 10**4)
    plt.legend(loc="upper left")
    plt.savefig('figs/ga_complexity_fitness_zoomed.png')
    plt.show()
