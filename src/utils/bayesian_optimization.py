from .qaoa_pulser_QPU import *
from .gaussian_process import *
import numpy as np
import time
import datetime
from ._differentialevolution import DifferentialEvolutionSolver
import pandas as pd
import os
import ast


class Bayesian_optimization():

    def __init__(self,
                 depth,
                 type_of_graph,
                 lattice_spacing,
                 nwarmup,
                 nbayes,
                 kernel_choice,
                 shots,
                 discard_percentage,
                 seed,
                 verbose_,
                 *args, 
                 **kwargs):
        '''Class Bayesian_optimization creates an instance of qaoa and of the
        gaussian process.
        
        init_training() creates the first nwarmup points
        run_optimization() calls iteratively the GP, proposes a new point, 
                           evaluate its energy with the QAOA class
                           
        PARAMETERS
        -----------
        depth: int
        type_of_graph: choice of graph
        lattice_spacing: int
        nwarmup: number of initial points (default 10)
        nbayes: number of optimization steps on top of nwarmup (default 100)
        kernel_choice: type of kernel for GP (matern, RBF etc..)
        shots: no. of shots
        seed: int
        verbose_: print during training or not 
        '''
        
        self.depth = depth
        self.nwarmup = nwarmup
        self.nbayes = nbayes
        self.shots = shots
        self.discard_percentage = discard_percentage
        self.seed = seed
        self.type_of_graph = type_of_graph
        angles_bounds = self.define_angles_boundaries(depth)
                
        ### CREATE QAOA
        self.qaoa = qaoa_pulser(depth, 
                                angles_bounds,
                                type_of_graph, 
                                lattice_spacing,
                                shots,
                                discard_percentage,
                                seed,
                                )

        ### CREATE GP 
        alpha = 10**(-6) #1/np.sqrt(self.shots)
        self.gp = MyGaussianProcessRegressor(depth = depth, 
                                             angles_bounds = angles_bounds,
                                             kernel_choice = kernel_choice,
                                             alpha = alpha,
                                             seed = seed)
        
        ### Training parameters
        self.kernel_matrices = []
        self.likelihood_landscapes = []
        self.final_states = []
        
        
    def define_angles_boundaries(self, depth):
        if depth == 1:
            angle_bounds = [
                            [100, 800] for _ in range(depth*2)
                            ]
        elif depth == 2:
            angle_bounds = [
                            [100, 800] for _ in range(depth*2)
                            ]
        elif depth == 3:
            angle_bounds = [
                            [100, 650] for _ in range(depth*2)
                            ]
        else:
            angle_bounds = [
                            [100, 500] for _ in range(depth*2)
                            ]
            
        return np.array(angle_bounds)
    
    def restrict_upper_angles_bounds(self, decrease):
        '''Called when a sequence of duration >4000ns is proposed to
            decrease the boundaries
        '''
        
        current_bounds = self.gp.angles_bounds
        new_bounds = current_bounds
        new_bounds[:, 1] -= decrease
        if new_bounds[0, 1] < 300:
            print('Cannot restrict bounds anymore')
            return
        else:
            self.gp.angles_bounds = new_bounds
            self.qaoa.angles_bounds = new_bounds
            print('restricted bounds to:', new_bounds)
           
    def save_info(self):
        self.file_name = (f'p={self.depth}_'
                         +f'{self.type_of_graph}_'
                         +f'shots_{self.shots}_'
                         +f'seed_{self.seed}')
        if self.discard_percentage > 0:
            self.file_name += f'_{self.discard_percentage}'
                            
        self.folder_name = 'output/'
        self.path_to_save = self.folder_name + self.file_name + '.csv'
        
        self.data_names = ['iter',
                          'point',
                          'energy_sampled',
                          'approximation_ratio', 
                          'energy_best',
                          'variance_sampled', 
                          'fidelity_sampled',
                          'fidelity_best', 
                          'ratio_solution',
                          'ratio_solution_best', 
                          'corr_length', 
                          'const_kernel',
                          'noise_level',
                          'std_energies', 
                          'average_distances', 
                          'n_iterations', 
                          'time_opt_bayes', 
                          'time_qaoa', 
                          'time_opt_kernel', 
                          'time_step',
                          'sampled_state'
                          ]
        self.data_header = " ".join(["{:>7} ".format(i) for i in self.data_names])
               
    def init_training(self):
        '''Selects self.nwarmup random points and fits the GP to them and starts
        saving data
        '''
        print('BAYESIAN OPTIMIZATION for QAOA')
        print(f'N steps = {self.nbayes}')
        print(f'N starting point = {self.nwarmup}')
        print(f'Depth = {self.depth}')
        print(f'Graph = {self.type_of_graph}')
        print(f'N shots = {self.shots}')
        X_train, y_train, data_train = self.qaoa.generate_random_points(self.nwarmup)
        self.gp.fit(X_train, y_train)
        
        ### The rest is just saving data ###
        
        kernel_params = np.exp(self.gp.kernel_.theta)
        self.data_ = []
        energy_best = np.max([data_train[i]['energy_sampled'] for i in range(len(X_train))])
        fidelity_best = np.max([data_train[i]['fidelity_sampled'] for i in range(len(X_train))])
        solution_ratio_best = np.max([data_train[i]['solution_ratio'] for i in range(len(X_train))])
        for i, x in enumerate(X_train):
            self.data_.append(
                              (i,
                                x, 
                                data_train[i]['energy_sampled'], 
                                data_train[i]['approximation_ratio'],
                                energy_best,
                                data_train[i]['variance_sampled'], 
                                data_train[i]['fidelity_sampled'],
                                fidelity_best,
                                data_train[i]['solution_ratio'],
                                solution_ratio_best,
                                kernel_params[0], 
                                kernel_params[1], 
                                kernel_params[2],
                                0, 0, 0, 0, 0, 0, 0,
                                data_train[i]['sampled_state']
                               )
                             )
        self.data_ = pd.DataFrame(data = self.data_, columns = self.data_names)
        self.data_.index.name='ITER'
        self.data_.to_csv(self.path_to_save)
               
    def acq_func(self, x):
        '''Calculate the acquisition function for the BO. Most costly part of
        the algorithm
        '''
        
        #check if acq_func is being evaluated on one point (needs reshaping) or many
        if isinstance(x[0], float):
            x = np.reshape(x, (1, -1))
        f_x, sigma_x = self.gp.predict(x, return_std=True) 
        f_prime = self.gp.y_best #current best value
        
        #Ndtr is a particular routing in scipy that computes the CDF in half the time
        cdf = ndtr((f_prime - f_x)/sigma_x)
        pdf = 1/(sigma_x*np.sqrt(2*np.pi)) * np.exp(-((f_prime -f_x)**2)/(2*sigma_x**2))
        alpha_function = (f_prime - f_x) * cdf + sigma_x * pdf
        
        return alpha_function
              
    def acq_func_maximize(self, x):
    
        return (-1)*self.acq_func(x)
        
    def bayesian_opt_step(self, init_pos = None):
        '''Performs the maximization of the acquisition function with the
        differential evolution algorithm
        '''
        
        samples = []
        acqfunvalues = []

        #callback to save progress data
        def callbackF(Xi, convergence):
            samples.append(Xi.tolist())
            acqfunvalues.append(self.acq_func(Xi, 1)[0])

        repeat = True
        with DifferentialEvolutionSolver(self.acq_func_maximize,
                                         bounds = [(0,1), (0,1)]*self.depth,
                                         callback = None,
                                         maxiter = 100*self.depth,
                                         popsize = 15,
                                         tol = .001,
                                         dist_tol = DEFAULT_PARAMS['distance_conv_tol'],
                                         seed = DEFAULT_PARAMS['seed']
                                         ) as diff_evol:
            results,average_norm_distance_vectors, std_population_energy, conv_flag = diff_evol.solve()
            next_point = results.x
        
            next_point = self.gp.scale_up(next_point)
                
        return next_point, results.nit, average_norm_distance_vectors, std_population_energy

    def check_proposed_point(self, point):
        X_ = self.gp.get_X()
        
        if point in X_:
            return False
        else:
            return True
            
    def run_optimization(self, load_data = None):
        '''Runs the whole optimization loop after initialization.
        The loop lasts for self.nbayes iterations.
        Proposed solutions at every step are checked if sum(sequence) <4000ns
        '''
        if load_data!=None:
            self.data_ = pd.read_csv(self.folder_name + load_data)
            
            X_train = [ast.literal_eval(i) for i in self.data_['point']]
            y_train = list(self.data_['energy_sampled'])
            self.gp.fit(X_train, y_train)
            previous_steps = len(X_train) - self.nwarmup
            fidelity_best = np.max(self.data_['fidelity_best'])
            solution_ratio_best = np.max(self.data_['ratio_solution_best'])
            iterations = self.nbayes - previous_steps 
            
            if iterations <= 0:
                raise Exception(f'You set the new number of steps to {self.nbayes} '
                                +f'but loaded a training with already {previous_steps} steps. '
                                +f'Make sure to set --nbayes to a number greater than {previous_steps}')
            print('LOADING DATA ...')
            print(f'Loaded data has {previous_steps} steps.')
            print(f'Restarting from step {previous_steps+1}')
            print('BAYESIAN OPTIMIZATION for QAOA')
            start_ = previous_steps
        else:
            start_ = 0
            iterations = self.nbayes
            fidelity_best = 0
            solution_ratio_best = 0
        
        print('Training ...')
        for i in range(start_, self.nbayes):
            start_time = time.time()
            
            #### BAYES OPT ####          
            counter = 0
            repeat = True
            while repeat and counter < 5:
                next_point, n_it, avg_sqr_distances, std_pop_energy = self.bayesian_opt_step()
                next_point = [int(j) for j in next_point]
                if sum(next_point)>(4000 - Q_DEVICE_PARAMS['first_pulse_duration']):
                    self.restrict_upper_angles_bounds(decrease = 50)
                    repeat = True
                    counter += 1
                else:
                    repeat = False
            bayes_time = time.time() - start_time
            
            ### QAOA on new point ###
            qaoa_results = self.qaoa.apply_qaoa(next_point)
            energy_sampled = qaoa_results['energy_sampled']
            y_next_point = energy_sampled
            best_point, energy_best, where_ = self.gp.get_best_point()
            fidelity_best = np.max((fidelity_best, 
                                        qaoa_results['fidelity_sampled']))
            solution_ratio_best = np.max((solution_ratio_best,
                                            qaoa_results['solution_ratio']))
            qaoa_time = time.time() - start_time - bayes_time
            
            
            ### FIT GP TO THE NEW POINT
            self.gp.fit(next_point, y_next_point)
            constant_kernel, corr_length, noise_level = np.exp(self.gp.kernel_.theta)

            kernel_time = time.time() - start_time - qaoa_time - bayes_time
            step_time = time.time() - start_time
            
            
            ###SAVING DATA ###
            new_data = pd.DataFrame(data = 
                        [[
                         i,
                         next_point,  
                         qaoa_results['energy_sampled'], 
                         qaoa_results['approximation_ratio'], 
                         energy_best,
                         qaoa_results['variance_sampled'], 
                         qaoa_results['fidelity_sampled'],
                         fidelity_best,
                         qaoa_results['solution_ratio'], 
                         solution_ratio_best,
                         corr_length, 
                         constant_kernel, 
                         noise_level,
                         std_pop_energy, 
                         avg_sqr_distances, 
                         n_it, 
                         bayes_time, 
                         qaoa_time, 
                         kernel_time, 
                         step_time,
                         qaoa_results['sampled_state']
                         ]],
                         columns = self.data_names
                        )
            self.data_= pd.concat([self.data_, new_data], ignore_index=True)
            
            print(f'iteration: {i +1}/{self.nbayes}  {next_point}'
                   f' (E - E_0)/E_0: {1 - y_next_point/self.qaoa.solution_energy}'
                   ' en: {}, fid: {}, ratio: {}'.format(y_next_point, qaoa_results['fidelity_sampled'], qaoa_results['solution_ratio'])
                   )
                    
            
           # df = pd.DataFrame(data = self.data_, columns = self.data_names)
           # df.to_csv(self.path_to_save)
            
            self.data_.to_csv(self.path_to_save, index=False)
            
        best_x, best_y, where = self.gp.get_best_point()
        # print(self.data_)
#         print(where)
#         last_point = pd.DataFrame(self.data_[where])
#         print(last_point)
#         self.data_ = pd.concat([self.data_,last_point], ignore_index=True)
        #self.data_.append(self.data_.iloc[where])
        #df = pd.DataFrame(data = self.data_, columns = self.data_names)
        #df.to_csv(self.path_to_save)
        #self.data_.to_csv(self.path_to_save)
        
        print('Best point: ' , self.data_.iloc[where])
