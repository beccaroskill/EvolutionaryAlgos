from vpython import scene, vector, mag, norm, proj, color, \
                    cylinder, box, sphere
from PIL import Image, ImageGrab
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import numpy.random as random
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool
import math
import csv
import copy

MU = 1
SHADOW_COLOR = vector(0.7, 0.7, 0.7)
SHADOW_HEIGHT = 10**(-5)
FREQ = 100
N_RODS = 72

class Mass:
    
    G = vector(0, -9.81, 0)
    DEFAULT_DAMPEN = 1
    
    def __init__(self, mass, pos, r, rotation, vis=True,
                 dampen=DEFAULT_DAMPEN, shadow=False):
        self.dampen = dampen
        self.r = r
        self.pos = pos
        self.mass = mass
        self.v = vector(0, 0, 0)
        self.a = Mass.G
        if vis:
            theta, axis, origin = rotation
            self.sphere = sphere(color = color.green, radius=r)
            self.sphere.pos = pos
            self.sphere.mass = mass
            self.sphere.rotate(angle=theta, axis=axis, origin=origin)
            self.pos = self.sphere.pos
        else:
            self.sphere = None
        if vis and shadow:
            self.shadow = cylinder(color=SHADOW_COLOR, radius=r, 
                                   axis=vector(0, 1, 0), length=SHADOW_HEIGHT,
                                   pos=vector(self.pos.x, 0, self.pos.z))
        else:
            self.shadow = None
        
    def update_a(self, Fs):
        a = vector(0,0,0)
        for F in Fs:
            a += F / self.mass
        self.a = a
    
    def update_mass(self, mass):
        self.mass = mass    
        if self.sphere:
            self.sphere.mass = mass    
        
    def update_pos(self, pos):
        self.pos = pos
        if self.sphere:
            self.sphere.pos = pos
            if self.shadow:
                self.shadow.pos.x = pos.x
                self.shadow.pos.z = pos.z
            
    def update_color(self, color):
        if self.sphere:
            self.sphere.color = color
        
    def time_step(self, dt):
        self.v =  self.dampen * (self.v + self.a * dt)
        self.update_pos(self.pos + self.v * dt)

class MassLink:
    
    DEFAULT_K = 1000000
    DEFAULT_R = 0.005

    
    def __init__(self, mass1, mass2, cube_indices=None, vis=True,
                 k=DEFAULT_K, r=DEFAULT_R, shadow=False):
        i_mass1, mass1 = mass1
        i_mass2, mass2 = mass2
        self.mass_indices = (i_mass1, i_mass2)
        self.L_rest = mag(mass2.pos - mass1.pos)
        self.k = k
        self.pos = mass1.pos
        self.axis = mass2.pos - mass1.pos
        self.length = mag(mass2.pos - mass1.pos)
        if vis:
            self.rod = cylinder(pos=self.pos, axis=self.axis, 
                                length=self.length, radius=r)
            self.cmap = mpl.cm.bwr
        else:
            self.rod = None
        if vis and shadow:
            self.shadow = self.get_shadow()
        else:
            self.shadow = None
        
    def get_shadow(self):
        if self.rod:
            v_rod = self.rod.axis
            v_proj = vector(v_rod.x, 0, v_rod.z)
            len_proj = mag(v_proj)
            dimensions = vector(len_proj, 0.0001, self.rod.radius)
            position = vector(self.rod.pos.x + self.rod.axis.x/2, 0, 
                                   self.rod.pos.z + self.rod.axis.z/2)
            return box(color=SHADOW_COLOR, size=dimensions, axis=v_proj,
                       pos=position)
        else:
            return None
    
    def update_shadow(self):
        if self.rod:
            v_rod = self.rod.axis
            v_proj = vector(v_rod.x, 0, v_rod.z)
            len_proj = mag(v_proj)
            dimensions = vector(len_proj, SHADOW_HEIGHT, self.rod.radius)
            self.shadow.size=dimensions
            self.shadow.axis=v_proj
            self.shadow.pos=vector(self.rod.pos.x + self.rod.axis.x/2 + self.rod.radius/2, 
                                   0, 
                                   self.rod.pos.z + self.rod.axis.z/2 + self.rod.radius/2)
    
    def update_link(self, mass1, mass2):
        self.pos = mass1.pos
        self.axis = mass2.pos - mass1.pos
        self.length = mag(mass2.pos - mass1.pos)
        if self.rod:
            self.rod.pos = self.pos
            self.rod.axis = self.axis
            self.rod.length = self.length
            if self.shadow:
                self.update_shadow()
        
    def get_force(self, mass):
        dL = self.L_rest - self.length 
        dL_norm = 10**8 * dL + 0.5
        if self.rod:
            self.rod.color = vector(*self.cmap(dL_norm)[:3])
        # Compression:
        if dL < 0:
            src_to_dest = 1 if self.pos == mass.pos else -1
        # Tension:
        elif dL > 0:
            src_to_dest = -1 if self.pos == mass.pos else 1
        else:
            src_to_dest = 1
        F_dir = norm(self.axis) * src_to_dest
        F_mag = self.k * abs(dL)
        return F_dir * F_mag
    
    def get_potential(self, mass):
        dL = self.L_rest - self.length 
        F = self.get_force(mass)
        F_mag = mag(F)
        return (1/2) * F_mag * dL


class Speciman:
    
    def __init__(self, breathe_params):
        self.breathe_params = breathe_params
        
    def evaluate(self, sim_time):
        start_p, end_p, T = self.simulate("shadow", vis=False, save_gif=False, 
                                          simulation_time=sim_time, plot_energy=False, drop_height=0.02)
        speed = mag(end_p - start_p) / T
        return speed
                  
    def simulate(self, name, vis=True, save_gif=True, simulation_time=0.05, 
                 plot_energy=True, drop_height=0.1, shadow=True):
    
        floor_y = 0
        floor_w = 4
        floor_h = 0.5
        floor_k = 200000
        if vis:
            wallB = box (pos=vector(0, -floor_h/2 + floor_y, 0), 
                         size=vector(-floor_w/2, floor_h, -floor_w/2),  color=color.white)
        
        scene.camera.pos = vector(0.12, 0.324885, 1.17934)
        
        cube_ps = [vector(0, drop_height, 0.7),
                   vector(0.2, drop_height, 0.7),]   
        
        masses = []
        rods = []
        for cube_p in cube_ps:
            mass_r = 0.01
            cube_m = 0.8
            cube_l = 0.1
            cube_rotation = (0, vector(0, 0, 0), cube_p)
    
            masses += [Mass(cube_m/6, cube_p, mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(0, cube_l, 0), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(0, 0, cube_l), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(0, cube_l, cube_l), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(cube_l, 0, cube_l), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(cube_l, cube_l, cube_l), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(cube_l, 0, 0), mass_r, cube_rotation, vis=vis),
                      Mass(cube_m/6, cube_p + vector(cube_l, cube_l, 0), mass_r, cube_rotation, vis=vis)]
            
        intercube_links = [(i, j) for i in range(8) for j in range(8,16) if i>3 and j<12]
        
        for i, mass1 in enumerate(masses):
            for j, mass2 in enumerate(masses[i+1:]):
                if (i < 8 and j+i+1 < 8) or (i >= 8 and j+i+1 >= 8) \
                    or (i, j+i+1) in intercube_links:
                    if (i, j+i+1) in intercube_links:
                        mass1.update_color(vector(1, 0, 0))
                        mass2.update_color(vector(1, 0, 0))
                    rod = MassLink((i, mass1), (j+i+1, mass2), vis=vis)
                    rods += [rod]
    
        Ls_rest = [rod.L_rest for rod in rods]
        cubes = [(masses, rods, Ls_rest)]
        
        dt = 0.0001
        T = 0
        
        for cube_i, cube in enumerate(cubes):
            masses, rods, Ls_rest = cube
            start_x, start_z = [0, 0]
            for mass in masses:
                start_x += mass.pos.x
                start_z += mass.pos.z
            start_p = vector(start_x/len(masses), 0, start_z/len(masses))
        
        while T < simulation_time:
            for cube_i, cube in enumerate(cubes):
                masses, rods, Ls_rest = cube
                for i, mass in enumerate(masses):
                    default_Fs = [Mass.G * mass.mass]
                    Fs = []
                    for rod in rods:
                        if i in rod.mass_indices:
                            Fs += [rod.get_force(mass)]
                            # if plot_energy:
                            #     PE += rod.get_potential(mass)
                    if (mass.pos.y-mass.r) < floor_y:
                        # floor spring
                        F_n = vector(0, floor_k * (floor_y - (mass.pos.y-mass.r)), 0)
                        Fs += [F_n]
                        # floor friction
                        v_floor = vector(-mass.v.x, 0, -mass.v.z)
                        Fs += [mag(F_n) * MU * norm(v_floor)]
                    Fs += default_Fs
                    mass.update_a(Fs)
                    mass.time_step(dt)
                for L_rest, params, rod in zip(Ls_rest, self.breathe_params, rods):
                    k, b, c = params
                    rod.update_link(*[masses[i] for i in rod.mass_indices])
                    rod.L_rest = L_rest + b * math.sin(FREQ * T + c) 
                    rod.k = k
            T += dt

        for cube_i, cube in enumerate(cubes):
            masses, rods, Ls_rest = cube
            end_x, end_z = [0, 0]
            for mass in masses:
                end_x += mass.pos.x
                end_z += mass.pos.z
                if mass.sphere:
                    mass.sphere.visible = False    
            for rod in rods:
                if rod.rod:
                    rod.rod.visible = False    
            end_p = vector(end_x/len(masses), 0, end_z/len(masses))
        return [start_p, end_p, T]

class Search:

    # min, max of uniform distribution for number of operations
    default_k_dist = [500000, 1500000]
    
    # min, max of uniform distribution for coefficients
    default_b_dist = [0, 0.025]
    
    # probability of variable (vs coefficient)
    default_c_dist = [0, 6.28]

    def __init__(self, k_dist=None, b_dist=None, c_dist=None,
                 var_ratio=None, operator_weights=None):
        if k_dist is None:
            self.k_dist = Search.default_k_dist     
        else:
            self.k_dist = k_dist
        if b_dist is None:
            self.b_dist = Search.default_b_dist     
        else:
            self.b_dist = b_dist
        if c_dist is None:
            self.c_dist = Search.default_c_dist     
        else:
            self.c_dist = c_dist
    
    def get_random_speciman(self):    
        breathe_params = [(random.uniform(*self.k_dist),
                           random.uniform(*self.b_dist),
                           random.uniform(*self.c_dist)) for i in range(N_RODS)]
        return Speciman(breathe_params)

    def get_mutation(self, speciman):   
        breathe_params = copy.deepcopy(speciman.breathe_params)
        mutate_pct = 0.1
        for i in range(len(breathe_params)):
            if random.uniform(0, 1) < mutate_pct:
                new_params = (random.uniform(*self.k_dist), 
                              random.uniform(*self.b_dist),
                              random.uniform(*self.c_dist)) 
                breathe_params[i] = new_params
      #  print('Difference:', sum([1 for i in range(3) \
      #                              for j in range(len(breathe_params)) \
      #                              if breathe_params[j][i] != speciman.breathe_params[j][i]]))
        return Speciman(breathe_params)
            
    def get_crossover(self, speciman_a, speciman_b):
        breathe_params_a = copy.deepcopy(speciman_a.breathe_params)
        breathe_params_b = copy.deepcopy(speciman_b.breathe_params)
        crossover_pts = sorted([int(random.uniform(0, 1) * N_RODS), 
                                int(random.uniform(0, 1) * N_RODS)])
        breathe_params = breathe_params_a[:crossover_pts[0]] + \
                         breathe_params_b[crossover_pts[0]:crossover_pts[1]] + \
                         breathe_params_a[crossover_pts[1]:]
        return Speciman(breathe_params)
             
       
    def reproduce(self, specimen, n_offspring):
        offspring = []
        while len(offspring) < n_offspring:
            speciman_a = random.choice(specimen)
            speciman_b = random.choice(specimen)
            speciman_crossed = self.get_crossover(speciman_a, speciman_b)
            speciman_mutated = self.get_mutation(speciman_crossed)
            offspring += [speciman_mutated]
        return offspring + specimen
    
    def run_random(self, n_trials, show_output=True):
        if show_output:
            print ('Random Search with', n_trials, 'trials')
        
        # Prep data storage for trials
        trials = range(n_trials + 1)
        best_scores = [-float('inf')]
        best_specimen = [None]
        best_speciman = None
        
        for i in range(n_trials):
            if show_output and i % (n_trials / 100) == 0:
                print ('Trial', i, 'of', n_trials)
                print ('Best score', round(best_scores[-1], 2))
                
            # Get a random expression as a heap, evaluate against data
            speciman = self.get_random_speciman()
            score = speciman.evaluate()
            
            # Update best score
            if score > best_scores[-1]:
                best_scores += [score]
                best_specimen += [speciman]
                best_speciman = speciman
            else:
                best_scores += [best_scores[-1]]
                best_specimen += [None]
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        best_specimen[-1] = best_speciman
        return (trials_df, best_specimen)
    
    def run_random_parallel(self, data, n_trials, num_nodes=None, plot=True):
        with ProcessPool(nodes=num_nodes) as pool:
            results = list(tqdm.tqdm(pool.imap(self.run_random, 
                                              [data for i in range(n_trials)], 
                                              [1 for i in range(n_trials)],
                                              [False for i in range(n_trials)],
                                              [False for i in range(n_trials)]), 
                                     total=n_trials))
            
        trial_dfs = [trial_df for trial_df, _ in results]
        specimen = [trial_specimen for _, trial_specimen in results]
        trials = range(n_trials + 1)
        best_scores = [float('inf')]
        best_specimen = [None]
        best_speciman = None
        
        for i in range(n_trials):
            trial_df = trial_dfs[i]
            trial_speciman = specimen[i][-1]
            trial_score = trial_df['best_scores'].to_list()[-1]
            if trial_score > best_scores[-1]:
                best_scores += [trial_score]
                best_specimen += [trial_speciman]
                best_speciman = trial_speciman
            else:
                best_scores += [best_scores[-1]]
                best_specimen += [None]

        # Plot best and worst path found over trials
        if plot:
            plt.figure(figsize=(6, 6))
            VisualizeSearch.plot_f(best_speciman, data)
            plt.show()        
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        best_specimen[-1] = best_speciman
        return (trials_df, best_specimen)
    
    def run_rmhc(self, n_trials, restart=None, plot=True, show_output=True):
        if show_output:
            print ('RMHC Search with', n_trials, 'trials')
        
        # Prep data storage for trials
        trials = range(n_trials + 1)
        best_scores = [-float('inf')]
        best_specimen = [None]
        best_speciman = None
        
        for i in range(n_trials):
            if show_output and i % (n_trials / 100) == 0:
                print ('Trial', i, 'of', n_trials)
                print ('Best score', round(best_scores[-1], 2))
                
            # Get a random expression as a heap, evaluate against data
            if best_speciman is None or restart and i % restart == 0:
                speciman = self.get_random_speciman()
            else:
                speciman = self.get_mutation(best_speciman)  
            score = speciman.evaluate()
            
            # Update best score
            if score > best_scores[-1]:
                print(score)
                best_scores += [score]
                best_specimen += [speciman]
                best_speciman = speciman
            else:
                best_scores += [best_scores[-1]]
                best_specimen += [None]
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        best_specimen[-1] = best_speciman
        return (trials_df, best_specimen)
    
    def run_rmhc_parallel(self, data, n_trials, restart=None, 
                          num_nodes=4, plot=True):
        # Prep params
        num_batches = int(n_trials / restart)
        batch_leftover = n_trials % restart
        pool_n_trials = [restart for i in range(num_batches)] 
        pool_n_trials += [batch_leftover] if batch_leftover else []
        if batch_leftover:
            num_batches += 1
        pool_params = [[data for i in range(num_batches)],   # dataset
                       pool_n_trials,                        # trial nums
                       [None for i in range(num_batches)],   # no restart
                       [False for i in range(num_batches)],  # suppress plotting
                       [False for i in range(num_batches)]]  # suppress output
        
        # Run parallel processes
        with ProcessPool(nodes=num_nodes) as pool:
            results = list(tqdm.tqdm(pool.imap(self.run_rmhc, *pool_params), 
                                total=num_batches))

        trial_dfs = [trial_df for trial_df, _ in results]
        trial_specimen = [trial_speciman for _, trial_speciman in results]
        all_trials = range(n_trials + num_batches + 1)
        all_best_scores = [float('inf')]
        all_best_specimen = [None]
        best_speciman = None
        
        for i in range(num_batches):
            df = trial_dfs[i]
            specimen = trial_specimen[i]
            best_scores = df['best_scores']
            for j, score in enumerate(best_scores):
                if score < all_best_scores[-1]:
                    all_best_scores += [score]
                    all_best_specimen += [specimen[j]]
                    best_speciman = specimen[j]
                else:
                    all_best_scores += [all_best_scores[-1]]
                    all_best_specimen += [None]
                    
        # Plot best path found over trials
        if plot:
            plt.figure(figsize=(6, 6))
            VisualizeSearch.plot_f(best_speciman, data)
            plt.show()
                
        # Compile data
        trials_df = pd.DataFrame({'trial': all_trials, 
                                  'best_scores': all_best_scores})
        all_best_specimen[-1] = best_speciman
        return (trials_df, all_best_specimen)
    
    def run_ga_parallel(self, n_trials, n_pop=100, top_k=30, num_nodes=4, plot=True,
                        parallel=True, trial_name=None, sim_time=1):
        # Prep data storage for trials
        num_gens = int(n_trials / n_pop)
        pop_specimen = [self.get_random_speciman() for i in range(n_pop)]
        trials = range(num_gens*n_pop + 1)
        best_scores = [-float('inf')]
        best_specimen = [None]
        best_speciman = None
        
        dots = []
        with ProcessPool(nodes=num_nodes) as pool:
            for i in range(num_gens):
                # Evaluate population
                if parallel:
                    pop_scores = pool.map(Speciman.evaluate, pop_specimen, 
                                          [sim_time for i in range(len(pop_specimen))])
                else:
                    pop_scores = [p.evaluate(sim_time) for p in pop_specimen] # unparallelize
                # Run to get dot plot data
                for score in pop_scores:
                    dots.append([i, score])
                # Update best scores
                for j, score in enumerate(pop_scores):
                    if score > best_scores[-1]:
                        best_scores += [score]
                        best_specimen += [pop_specimen[j]]
                        best_speciman = pop_specimen[j]
                    else:
                        best_scores += [best_scores[-1]]
                        best_specimen += [None]
                # Selection
                ordering = np.argsort(pop_scores).astype(int)
                ordering = ordering[::-1]
                top_specimen = [speciman for i, speciman in enumerate(pop_specimen)
                                if i in ordering[:top_k]]
                # Reproduction
                pop_specimen = self.reproduce(top_specimen, n_pop-top_k)
                print(i+1, 'of', num_gens)
                print('Best score:', best_scores[-1])
                print('Average score:', np.mean(pop_scores))
                
        # Plot best and worst path found over trials
        # if plot:
        #     plt.figure(figsize=(6, 6))
        #     VisualizeSearch.plot_f(best_speciman, data)
        #     plt.show()
        
        # Compile data
        trials_df = pd.DataFrame({'trial': trials, 
                                  'best_scores': best_scores})
        best_specimen[-1] = best_speciman
        
        with open("results/dotplot/dots_i{}.csv".format(trial_name), mode="w") as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(['Index', 'Speed'])
            csvwriter.writerows(dots)

            
        return (trials_df, best_specimen)
               
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
    
    def plot_fitness(results_dir, batch_key, batch_label, experiment_name, 
                     fitness_f=None, xlim=None, ylim=None):
        dfs = []
        for f in os.listdir(results_dir):
            if 'csv' in f and batch_key in f:
                print(f)
                df = pd.read_csv(results_dir + f)
                dfs += [df]
    
        trials = dfs[0]['trial']
        if fitness_f:
            best_scores = fitness_f(np.mean([df['best_scores'] 
                                             for df in dfs], axis=0))
        else:
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
        df, best_specimen = get_frames_f(*get_frames_args)
        best_scores = df['best_scores']
        
        X = [x_val for x_val, _ in data]
        Y = [y_val for _, y_val in data]
        x_lim = (min(X), max(X))
        y_lim = (min(Y), max(Y))
        
        fig, ax = plt.subplots()
        ln, = plt.plot([], [], 'orange')
        global mse
        mse = float('inf')
        
        def init():
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            return ln,
        
        def update(i):
            global mse
            if best_specimen[i]:
                spec = best_specimen[i]
                Y_f = [spec.evaluate_at(x)[1] for x in X]
                plt.cla()
                ln.set_data(X, Y_f)
                ax.set_xlim(*x_lim)
                ax.set_ylim(*y_lim)
                plt.plot(X, Y, c='b')
                plt.plot(X, Y_f, c='orange')
                mse = round(best_scores[i], 2)
            title = 'Best fit function, mse={} (Trial {})'.format(mse, i+1)
            plt.title(title)
            return ln,
        
        anim = FuncAnimation(fig, update, frames=len(best_specimen),
                            init_func=init, blit=True, interval=10)
        writergif = PillowWriter(fps=10000) 
        anim.save(save_path, writer=writergif)
        plt.show()
    

