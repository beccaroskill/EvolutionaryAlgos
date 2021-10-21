from symbolic_regression import SearchAlgorithms, VisualizeSearch, load_dataset
from matplotlib import pyplot as plt

if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 100

    random_search = SearchAlgorithms()

    
    for i in range(1, 6):
        df, best_specimen = random_search.run_random_parallel(dataset, n_trials, 
                                                            num_nodes=None,
                                                            plot=True)
        results_subdir = 'results/random_ops6to64'
        df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
        expression_summary = '{}, MSE: {}'.format(best_specimen[-1].to_expr(),
                                                  df['best_scores'].to_list()[-1])
        with open('{}/n{}_i{}.txt'.format(results_subdir, n_trials, i), 'w') as f:
            f.write(expression_summary)
        print(expression_summary)
        plt.figure(figsize=(6, 6))
        VisualizeSearch.plot_f(best_specimen[-1], dataset)
        plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
        plt.show() 
    
    for i in range(1, 6):
        df, best_specimen = random_search.run_rmhc_parallel(dataset, n_trials, 
                                                            restart=int(n_trials/100), 
                                                            num_nodes=None,
                                                            plot=True)
        results_subdir = 'results/rmhc_100restarts_ops6to64'
        df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
        expression_summary = '{}, MSE: {}'.format(best_specimen[-1].to_expr(),
                                                  df['best_scores'].to_list()[-1])
        with open('{}/n{}_i{}.txt'.format(results_subdir, n_trials, i), 'w') as f:
            f.write(expression_summary)
        print(expression_summary)
        plt.figure(figsize=(6, 6))
        VisualizeSearch.plot_f(best_specimen[-1], dataset)
        plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
        plt.show() 


    # f = 'results/random/n10000_i4.csv'
    # df = pd.read_csv(f)
    # print(df['best_scores'].to_list()[-1])
    # expr = 'Mul(10, Pow(2, -1))'
    # expr = parse_expr(expr)
    # print(expr.evalf())