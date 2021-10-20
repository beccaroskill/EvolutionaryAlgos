from sympy.parsing.sympy_parser import parse_expr      
from collections import deque 
import math
import numpy as np 
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import pandas as pd
from pathos.multiprocessing import ProcessPool

class ExpressionHeap:
    
    valid_operators = ['Add', 'Mul', 'Sub', 'Div', 'sin', 'cos']
    all_operators = ['Add', 'Mul', 'Pow', 'Sub', 'Div', 'sin', 'cos']
    unary_operators = ['sin', 'cos']

    
    def __init__(self, expr=None, heap=[]):
        if expr:
            self.heap = ExpressionHeap.from_expr(expr)
        else:
            self.heap = heap
    
    def separate_str_args(args):
        comma_i = [i for i, x in enumerate(args) if x == ',']
        for i in comma_i:
            open_count = len([c for c in args[:i] if c == '('])
            closed_count = len([c for c in args[:i] if c == ')'])
            if open_count == closed_count:
                left = args[:i]
                right = args[i+1:].strip()
                return (left, right)
            
    def is_legal_operation(op, left, right):
        legal = True
        if op == 'Div' and isinstance(right, float) and right == 0:
            legal = False
        return legal
    
    def build_heap(expr, heap, parent_i):
        if ',' in expr:
            operator = expr[:expr.find('(')]
            heap[parent_i] = operator
            left, right = ExpressionHeap.separate_str_args(expr[expr.find('(')+1: expr.rfind(')')])
            heap = ExpressionHeap.build_heap(left, heap, 2*parent_i + 1)
            heap = ExpressionHeap.build_heap(right, heap, 2*parent_i + 2)
        else:
            # We might need to allocate more memory if tree is unbalanced
            if len(heap) <= parent_i:
                heap += [None] * (parent_i - len(heap) + 1)
            heap[parent_i] = expr
        return heap
    
    def from_expr(expr):
        num_ops = len([i for i, x in enumerate(expr) if x == ','])
        
        # Best case, our tree is balanced (we may need to allocate more later)
        heap = [None] * (2 * num_ops + 1)
        heap = ExpressionHeap.build_heap(expr, heap, 0)
        return heap
    
    def to_expr(self, include_sub_expr=False):
        arg_stack = deque() 
        str_expr = ''
        heap = self.heap.copy()
        sub_expr = {} if include_sub_expr else None
        
        # Deal with expressions that are just one term, no operators
        if len(heap) == 1:
            str_expr = str(heap[0])
            if include_sub_expr:
                sub_expr[0] = str_expr
                
        for i in range(len(heap)-1, 0, -1):
            x = heap[i]
            if x not in ExpressionHeap.all_operators:
                if arg_stack:
                    parent_i = math.floor((i - 1)/2)
                    operator = heap[parent_i]
                    left, right = [x, arg_stack.pop()]
                    if operator is not None and left is not None:
                        if operator == 'Sub':
                            str_expr = 'Add({}, Mul(-1, {}))'.format(left, right)
                        elif operator == 'Div':
                            str_expr = 'Mul({}, Pow({}, -1))'.format(left, right)
                        elif operator in ExpressionHeap.unary_operators:
                            str_expr = '{}({})'.format(operator, left)
                        else:
                            str_expr = '{}({}, {})'.format(operator, left, right)
                        if include_sub_expr:
                            sub_expr[parent_i] = str_expr
                        if i > 0:
                            heap[parent_i] = str_expr
                else:
                    arg_stack.append(x)
                    
        return (str_expr, sub_expr) if include_sub_expr else str_expr
    
    def evaluate_at(self, x_val):
        heap = self.heap.copy()
        arg_stack = deque() 

        # Deal with expressions that are just one term, no operators
        if len(heap) == 1:
            heap[0] = [heap[0] if heap[0] != 'x' else x_val]
                
        for i in range(len(heap)-1, 0, -1):
            x = heap[i]
            if x not in ExpressionHeap.all_operators:
                if arg_stack:
                    parent_i = math.floor((i - 1)/2)
                    operator = heap[parent_i]
                    left, right = [arg if arg != 'x' else x_val
                                   for arg in [x, arg_stack.pop()] ]
                    if operator is not None and left is not None:
                        if operator == 'Add':
                            parent_val = left + right                        
                        elif operator == 'Sub':
                            parent_val = left - right
                        elif operator == 'Mul':
                            parent_val = left * right
                        elif operator == 'Div':
                            try: 
                                parent_val = left / right
                            except ZeroDivisionError:
                                return (False, None)
                        elif operator == 'sin':
                            parent_val = math.sin(left)
                        elif operator == 'cos':
                            parent_val = math.cos(left)
                        else:
                            print('Operator unknown:', operator)
                        heap[parent_i] = parent_val
                else:
                    arg_stack.append(x)
        return (True, heap[0])
    
    def evaluate(self, data):
        sse = 0
        for x_val, y_val in data:
            real, f_val = self.evaluate_at(x_val)
            if not real:
                return (False, None)
            e_sq = (f_val - y_val)**2
            sse += e_sq
        mse = sse/len(data)
        return (True, mse)
                    

    def replace_subtree(self, subroot, subtree):
        # TODO - Expand heap if subtree requires it
        
        # TODO - Replace subtree
        # for i in (len(subtree) - 1 / 2):
        # For now, just assume replacing subtree with a constant
        self.heap[subroot] = subtree[0]
    
        # Remove extraneous children
        for i in range(len(self.heap)):
            parent_i = math.floor((i - 1)/2)
            if parent_i > 0 and self.heap[parent_i] not in ExpressionHeap.all_operators:
                self.heap[i] = None 
            elif parent_i == 0 and len(self.heap) == 3:
                self.heap = [ self.heap[0] ]

        expr = self.to_expr()
        self.heap = ExpressionHeap.from_expr(expr)
        
    def trim_heap(self, data, threshold=0.1):

        _, sub_expr = self.to_expr(include_sub_expr=True)
        for subroot in sub_expr:
            str_expr = sub_expr[subroot]
            expr = parse_expr(str_expr)
            f_vals = [expr.subs('x', x_val).evalf() for x_val, _ in data]
            var = np.var(f_vals)
            if var < threshold:
                mean = np.mean(f_vals)
                self.replace_subtree(subroot, [mean])
                
class VisualizeSearch:
    
    def plot_f(f_heap, data, mse=None, ax=None):
        str_expr = f_heap.to_expr()
        expr = parse_expr(str_expr)
        
        X = [x_val for x_val, _ in data]
        Y = [y_val for _, y_val in data]
        Y_f = []
        for x_val in X:
            y_val = expr.subs('x', x_val).evalf()
            Y_f += [y_val] 
        
        plt.plot(X, Y, label='Input data')
        plt.plot(X, Y_f, label=str(expr))
        title = 'Best fit function'
        mse_real, mse = f_heap.evaluate(data)
        if mse_real:
            title += '\nMSE: {}'.format(mse)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')
    
    def plot_trials(results_dir, batch_key, batch_label, experiment_name, xlim=None, ylim=None):
        dfs = []
        for f in os.listdir(results_dir):
            if 'csv' in f and batch_key in f:
                print(f)
                df = pd.read_csv(results_dir + f)
                dfs += [df]
    
        trials = dfs[0]['trial']
        best_scores = 10**2-np.mean([df['best_scores'] for df in dfs], axis=0)
        std_err_pts = np.logspace(0, np.log10(max(trials)), 10).astype(int)
        best_scores_err = [np.std([-df['best_scores'][i] for df in dfs]) 
                         for i in std_err_pts]
        plt.subplot(111)
        plt.plot(trials, best_scores, label=batch_label)
        best_scores_c = plt.gca().lines[-1].get_color()
        plt.errorbar(std_err_pts, [best_scores[i] for i in std_err_pts], 
                      yerr=best_scores_err, linestyle='', capsize=2, c=best_scores_c)
        plt.title(experiment_name)
        plt.xlabel('Trial')
        plt.xscale('log')
        plt.ylabel('Fitness')
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.legend(loc='lower right')
    
    def plot_animation(data, save_path, get_frames_f, get_frames_args):
        best_specimen = get_frames_f(*get_frames_args)
        
        X = [x_val for x_val, _ in data]
        Y = [y_val for _, y_val in data]
        x_lim = (min(X), max(X))
        y_lim = (min(Y), max(Y))
        
        fig, ax = plt.subplots()
        ln, = plt.plot([], [], 'orange')
        mse = float('inf')
        
        def init():
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            return ln,
        
        def update(i):
            global mse
            if best_specimen[i]:
                spec = best_specimen[i][0]
                Y_f = [spec.evaluate_at(x)[1] for x in X]
                plt.cla()
                ln.set_data(X, Y_f)
                ax.set_xlim(*x_lim)
                ax.set_ylim(*y_lim)
                plt.plot(X, Y, c='b')
                plt.plot(X, Y_f, c='orange')
                mse = round(best_specimen[i][1], 2)
            title = 'Best fit function, mse={} (Trial {})'.format(mse, i+1)
            plt.title(title)
            return ln,
        
        anim = FuncAnimation(fig, update, frames=len(best_specimen),
                            init_func=init, blit=True, interval=10)
        writergif = PillowWriter(fps=10000) 
        anim.save(save_path, writer=writergif)
        plt.show()


class SearchAlgorithms:
    
    # min, max of uniform distribution for number of operations
    default_n_ops_dist = [3, 6] 
    
    # min, max of uniform distribution for coefficients
    default_coef_dist = [-10, 10]
    
    # probability of variable (vs coefficient)
    default_var_ratio = 0.5 
    
    # distribution of operator probabilities
    default_operator_weights = [1/len(ExpressionHeap.valid_operators) 
                                for op in ExpressionHeap.valid_operators]
        
    def __init__(self, n_ops_dist=None, coef_dist=None,
                 var_ratio=None, operator_weights=None):
        
        if n_ops_dist is None:
            self.n_ops_dist = SearchAlgorithms.default_n_ops_dist               
        if coef_dist is None:
            self.coef_dist = SearchAlgorithms.default_coef_dist        
        if var_ratio is None:
            self.var_ratio = SearchAlgorithms.default_var_ratio
        if operator_weights is None:
            self.operator_weights = SearchAlgorithms.default_operator_weights
    
    def get_random_heap(self):
        n_ops = random.randint(*self.n_ops_dist)
        ops = random.choices(ExpressionHeap.valid_operators, 
                             weights=self.operator_weights, k=n_ops)
        heap = [None] * (n_ops * 2)
        
        parent_indices = [0]
        while ops:
            i = parent_indices.pop(0)
            op = ops.pop(0)
            heap[i] = op
            parent_indices += [2*i+1] 
            coef = round(random.uniform(*self.coef_dist), 2)
            swap = random.choice([True, False])
            left = 'x' if not swap else coef
            right = coef if not swap else 'x'
            while 2*i+2 >= len(heap):
                heap += [None]
            heap[2*i+1] = left
            heap[2*i+2] = right
        return ExpressionHeap(heap=heap)
    
    def change_coef(self, specimen):
        mutation = specimen.heap.copy()
        arg_indices = [i for i, x in enumerate(specimen.heap) 
                          if x and x != 'x' and x not in ExpressionHeap.valid_operators]  
        arg_i = random.choice(arg_indices)
        coef = round(random.uniform(*self.coef_dist), 2)
        mutation[arg_i] = coef
        return mutation
    
    def change_operator(self, specimen):
        mutation = specimen.heap.copy()
        op_indices = [i for i, x in enumerate(specimen.heap) 
                          if x and x in ExpressionHeap.valid_operators]        
        i = random.choice(op_indices)
        op = random.choice(ExpressionHeap.valid_operators)
        mutation[i] = op
        return mutation
    
    def add_subtree(self, specimen):
        mutation = specimen.heap.copy()
        arg_indices = [i for i, x in enumerate(specimen.heap) 
                          if x and x not in ExpressionHeap.valid_operators]       
        i = random.choice(arg_indices)
        op = random.choice(ExpressionHeap.valid_operators)
        mutation[i] = op
        coef = round(random.uniform(*self.coef_dist), 2)
        swap = random.choice([True, False])
        left = 'x' if not swap else coef
        right = coef if not swap else 'x'
        while 2*i+2 >= len(mutation):
            mutation += [None]
        mutation[2*i+1] = left
        mutation[2*i+2] = right
        return mutation
    
    def remove_subtree(self, specimen):
        mutation = specimen.heap.copy()
        op_indices = [i for i, x in enumerate(specimen.heap) 
                          if x and x in ExpressionHeap.valid_operators 
                          and i != 0]        
        i = random.choice(op_indices)
        coef = round(random.uniform(*self.coef_dist), 2)
        arg = random.choice([coef, 'x'])
        mutation[i] = arg
        children = [2*i+1, 2*i+2]
        while children:
            child_i = children.pop(0)
            if child_i < len(mutation):
                mutation[child_i] = None
                children += [2*child_i+1, 2*child_i+2]
        return mutation
    
    def get_mutation(self, specimen):    
        num_ops = len([i for i, x in enumerate(specimen.heap) 
                          if x and x in ExpressionHeap.valid_operators])
        if num_ops < self.n_ops_dist[0]:
            mutation_fs = [self.add_subtree]
        elif num_ops > self.n_ops_dist[1]:
            mutation_fs = [self.remove_subtree]
        else:
            mutation_fs = [self.change_coef, self.change_operator,
                           self.add_subtree, self.remove_subtree]
        mutation_f = random.choice(mutation_fs)
        mutation = mutation_f(specimen)
        return ExpressionHeap(heap=mutation)
            
    def run_random(self, data, n_trials, plot=True, show_output=True):
        if show_output:
            print ('Random Search with', n_trials, 'trials')
        
        # Prep data storage for trials
        trials = range(n_trials + 1)
        best_scores = [float('inf')]
        best_specimen = None
        
        for i in range(n_trials):
            if show_output and i % (n_trials / 10) == 0:
                print ('Trial', i, 'of', n_trials)
                print ('Best score', round(best_scores[-1], 2))
                
            # Get a random expression as a heap, evaluate against data
            real = False
            while not real:
                specimen = self.get_random_heap()
                real, score = specimen.evaluate(data)
            
            # Update best score
            if score < best_scores[-1]:
                best_scores += [score]
                best_specimen = specimen
            else:
                best_scores += [best_scores[-1]]
        
        # Plot best and worst path found over trials
        if plot:
            plt.figure(figsize=(6, 6))
            VisualizeSearch.plot_f(best_specimen, data)
            plt.show()
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        return (trials_df, best_specimen)
    
    def run_random_parallel(self, data, n_trials, plot=True):
        pool = ProcessPool(nodes=8)
        results = pool.map(self.run_random, [data for i in range(n_trials)], 
                            [1 for i in range(n_trials)],
                            [False for i in range(n_trials)],
                            [False for i in range(n_trials)])
        trial_dfs = [trial_df for trial_df, _ in results]
        specimen = [trial_specimen for _, trial_specimen in results]
        trials = range(n_trials + 1)
        best_scores = [float('inf')]
        for i in range(n_trials):
            trial_df = trial_dfs[i]
            trial_specimen = specimen[i]
            trial_score = trial_df['best_scores'].to_list()[-1]
            if trial_score < best_scores[-1]:
                best_scores += [trial_score]
                best_specimen = trial_specimen
            else:
                best_scores += [best_scores[-1]]
                
        # Plot best and worst path found over trials
        if plot:
            plt.figure(figsize=(6, 6))
            VisualizeSearch.plot_f(best_specimen, data)
            plt.show()        
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        return (trials_df, best_specimen)
    
    def run_rmhc(self, data, n_trials, plot=True, show_output=True):
        if show_output:
            print ('RMHC Search with', n_trials, 'trials')
        
        # Prep data storage for trials
        trials = range(n_trials + 1)
        best_scores = [float('inf')]
        best_specimen = None
        
        for i in range(n_trials):
            if show_output and i % (n_trials / 10) == 0:
                print ('Trial', i, 'of', n_trials)
                print ('Best score', round(best_scores[-1], 2))
                
            # Get a random expression as a heap, evaluate against data
            real = False
            while not real:
                if best_specimen:
                    specimen = self.get_mutation(best_specimen)  
                else:
                    specimen = self.get_random_heap()
                real, score = specimen.evaluate(data)
            
            # Update best score
            if score < best_scores[-1]:
                best_scores += [score]
                best_specimen = specimen
            else:
                best_scores += [best_scores[-1]]
        
        # Plot best and worst path found over trials
        if plot:
            plt.figure(figsize=(6, 6))
            VisualizeSearch.plot_f(best_specimen, data)
            plt.show()
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        return (trials_df, best_specimen)
    
    def get_rmhc_frames(self, data, n_trials, n_frames=500):

        # Prep data storage for trials
        frames = []
        best_scores = [float('inf')]
        best_specimen = None
        
        for i in range(n_trials):
            if i % (n_trials / 10) == 0:
                print ('Trial', i, 'of', n_trials)
                print ('Best score', round(best_scores[-1], 2))
            # Get a random expression as a heap, evaluate against data
            real = False
            while not real:
                if best_specimen:
                    specimen = self.get_mutation(best_specimen)  
                else:
                    specimen = self.get_random_heap()
                real, score = specimen.evaluate(data)
            
            # Update best score
            if score < best_scores[-1]:
                best_scores += [score]
                best_specimen = specimen
                frames += [(best_specimen, best_scores[-1])]
            else:
                best_scores += [best_scores[-1]]
                frames += [None]

        return frames
    
def load_dataset(path):
    # Load dataset from .txt file
    with open(path, 'r') as f:
        lines = f.readlines()
    return [[float(x) for x in line[:-1].split(',')] for line in lines]

if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 1000

    random_search = SearchAlgorithms()

    VisualizeSearch.plot_animation(dataset, 'figs/rmhc_animation_2.gif', 
                                   random_search.get_rmhc_frames, 
                                   [dataset, n_trials])
    
    # for i in range(1, 6):
    #     df, best_specimen = random_search.run_rmhc(dataset, n_trials, plot=True)
    #     df.to_csv('results/rmhc/n{}_i{}.csv'.format(n_trials, i))
    #     print(best_specimen.to_expr(), 'MSE', df['best_scores'].to_list()[-1])
    #     plt.figure(figsize=(6, 6))
    #     VisualizeSearch.plot_f(best_specimen, dataset)
    #     plt.show()        
        
    # VisualizeSearch.plot_trials('results/rmhc/', 'n{}'.format(n_trials), 
    #                             'rmhc', 'RMHC Search', ylim=(0,100))
    # VisualizeSearch.plot_trials('results/random/', 'n{}'.format(n_trials), 
    #                             'random', 'Random Search', ylim=(0,100))
    
    # f = 'results/random/n10000_i4.csv'
    # df = pd.read_csv(f)
    # print(df['best_scores'].to_list()[-1])
    # expr = 'Mul(10, Pow(2, -1))'
    # expr = parse_expr(expr)
    # print(expr.evalf())