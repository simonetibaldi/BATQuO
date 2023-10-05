from .qaoa_pulser import *
from .gaussian_process import *
import numpy as np
import time
import datetime
from ._differentialevolution import DifferentialEvolutionSolver
import pandas as pd
import os


class Bayesian_optimization():

    def __init__(self,
                 depth,
                 type_of_graph,
                 lattice_spacing,
                 quantum_noise,
                 nwarmup,
                 nbayes,
                 kernel_choice,
                 shots,
                 seed,
                 verbose_,
                 *args, 
                 **kwargs):
             
        self.depth = depth
        self.nwarmup = nwarmup
        self.nbayes = nbayes
        self.shots = shots
        self.seed = seed
        angles_bounds = self.define_angles_boundaries(depth)
                
        ### CREATE QAOA
        self.qaoa = qaoa_pulser(depth, 
                                angles_bounds,
                                type_of_graph, 
                                lattice_spacing,
                                shots,
                                seed,
                                quantum_noise)
        self.type_of_graph = type_of_graph

        ### CREATE GP 
        self.gp = MyGaussianProcessRegressor(depth = depth, 
                                             angles_bounds = angles_bounds,
                                             kernel_choice = kernel_choice,
                                             seed = seed)
        
        ### Training parameters
        self.kernel_matrices = []
        self.likelihood_landscapes = []
        self.final_states = []
        
        
    def define_angles_boundaries(self, depth):
        if depth == 1:
            angle_bounds = [
                            [200, 800] for _ in range(depth*2)
                            ]
        elif depth == 2:
            angle_bounds = [
                            [200, 800] for _ in range(depth*2)
                            ]
        elif depth == 3:
            angle_bounds = [
                            [200, 650] for _ in range(depth*2)
                            ]
        else:
            angle_bounds = [
                            [200, 500] for _ in range(depth*2)
                            ]
            
        return np.array(angle_bounds)
    
    def restrict_upper_angles_bounds(self, decrease):
        current_bounds = self.gp.angles_bounds
        new_bounds = current_bounds
        new_bounds[:, 1] -= decrease
        if new_bounds[0, 1] < 500:
            print('Cannot restrict bounds anymore')
            return
        else:
            self.gp.angles_bounds = new_bounds
            self.qaoa.angles_bounds = new_bounds
            print('restricted bounds to:', new_bounds)
      
        
    def angle_names_string(self):
        gamma_names = [f'GAMMA_{i}' for i in range(self.depth)]
        beta_names = [f'BETA_{i}' for i in range(self.depth)]
        
        angle_names = []
        for i in range(self.depth):
            angle_names.append(beta_names[i])
            angle_names.append(gamma_names[i])
            
        return angle_names
           
    def print_info(self):
        self.file_name = (f'p={self.depth}_'
                         +f'{self.type_of_graph}_'
                         +f'shots_{self.shots}_'
                         +f'_seed_{self.seed}')
                            
        self.folder_name = 'output/' + self.file_name + '/'
        os.makedirs(self.folder_name, exist_ok = True)
        angle_names = self.angle_names_string()
        
        self.data_names = [
                          'iter',
                          'point',
                          'energy_sampled',
                          'classical_solution',
                          'ratio_sampled_classical', 
                          'energy_exact',
                          'energy_gs',
                          'ratio_exact_gs',
                          'energy_best',
                          'variance_sampled', 
                          'variance_exact',
                          'fidelity_exact', 
                          'fidelity_sampled',
                          'fidelity_best', 
                          'ratio_solution',
                          'ratio_solution_best', 
                          'corr_length', 
                          'const_kernel',
                          'std_energies', 
                          'average_distances', 
                          'n_iterations', 
                          'time_opt_bayes', 
                          'time_qaoa', 
                          'time_opt_kernel', 
                          'time_step',
                          'doppler_detune',
                          'actual_pulse_parameters',
                          'bad_atoms',
                          'final_state'
                          ]
        self.data_header = " ".join(["{:>7} ".format(i) for i in self.data_names])
        
        self.info_file_name = self.folder_name + self.file_name + '_info.txt'
        with open(self.info_file_name, 'w') as f:
            f.write('BAYESIAN OPTIMIZATION of QAOA \n\n')
            self.qaoa.print_info_problem(f)
            
            f.write('QAOA PARAMETERS')
            f.write('\n-------------\n')
            self.qaoa.print_info_qaoa(f)
            
            f.write('\nGAUSSIAN PROCESS PARAMETERS')
            f.write('\n-------------\n')
            self.gp.print_info(f)
            
            f.write('\nBAYESIAN OPT PARAMETERS')
            f.write('\n-------------\n')
            f.write(f'Nwarmup points: {self.nwarmup} \n')
            f.write(f'Ntraining points: {self.nbayes}\n')
            f.write('FILE.DAT PARAMETERS:\n')
            print(self.data_names, file = f)
               
    def init_training(self, Nwarmup):
        X_train, y_train, data_train = self.qaoa.generate_random_points(Nwarmup)
        self.gp.fit(X_train, y_train)
        
        df = pd.DataFrame(np.column_stack((X_train, y_train)))
        print('### TRAIN DATA ###')
        print(df)
        print('\nKernel after training fit')
        print(self.gp.kernel_)
        print('\nStarting K')
        print(self.gp.get_covariance_matrix())
        
        kernel_params = np.exp(self.gp.kernel_.theta)
        self.data_ = []
        energy_best = np.max([data_train[i]['energy_sampled'] for i in range(len(X_train))])
        fidelity_best = np.max([data_train[i]['fidelity_sampled'] for i in range(len(X_train))])
        solution_ratio_best = np.max([data_train[i]['solution_ratio'] for i in range(len(X_train))])
        gs_en = self.qaoa.gs_en
        for i, x in enumerate(X_train):
            self.data_.append(
                              (i +1,
                               x , 
                               y_train[i], 
                                self.qaoa.solution_energy, 
                                1 - y_train[i]/ self.qaoa.solution_energy,
                                data_train[i]['energy_exact'],
                                gs_en, 
                                1 - data_train[i]['energy_exact']/gs_en,
                                energy_best,
                                data_train[i]['variance_sampled'], 
                                data_train[i]['variance_exact'],
                                data_train[i]['fidelity_sampled'],
                                data_train[i]['fidelity_exact'], 
                                fidelity_best,
                                data_train[i]['solution_ratio'],
                                solution_ratio_best,
                                kernel_params[0], 
                                kernel_params[1], 
                                0, 0, 0, 0, 0, 0, 0,
                                data_train[i]['doppler_detune'],
                                data_train[i]['actual_pulse_parameters'],
                                data_train[i]['bad_atoms'],
                                data_train[i]['final_state']
                                )
                             )
        self.data_file_name = self.file_name + '.dat'
        df = pd.DataFrame(data = self.data_, columns = self.data_names)
        df.to_pickle(
                    self.folder_name + self.file_name, 
                    )
               
    def acq_func(self, x):
        
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
            
    def run_optimization(self):
    
        fidelity_best = 0
        solution_ratio_best = 0
        simplified_data = []
        print('Training ...')
        for i in range(self.nbayes):
            start_time = time.time()
            
            #### BAYES OPT ####
            #next_point, n_it, avg_sqr_distances, std_pop_energy = self.bayesian_opt_step()
            #next_point = [int(i) for i in next_point]
            
            
            counter = 0
            repeat = True
            while repeat and counter < 5:
                next_point, n_it, avg_sqr_distances, std_pop_energy = self.bayesian_opt_step()
                next_point = [int(i) for i in next_point]
                if sum(next_point)>3500:
                    self.restrict_upper_angles_bounds(decrease = 100)
                    repeat = True
                    counter += 1
                else:
                    repeat = False
            # check_ = self.check_proposed_point(next_point)
#             if not check_:
#                 print(f'Found the same point twice {next_point} by the optimization')
#                 print('ending optimization')
#                 break
            
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
            
            
            self.gp.fit(next_point, y_next_point)
            #likelihood_landscape = self.gp.get_log_marginal_likelihood(show = False, 
            #                                                            save = False)
            #cov_matrix = self.gp.get_covariance_matrix()
            #optimization_samples = self.gp.kernel_opt_samples
            #self.likelihood_landscapes.append(likelihood_landscape)
            #self.kernel_matrices.append(cov_matrix.tolist())
            
            
            # np.save(self.folder_name + 'likelihoods', 
#                         self.likelihood_landscapes)  #saved in binary bc its is 3d
#             np.save(self.folder_name + 'cov_matrices', 
#                          np.array(self.kernel_matrices)) 
#             np.save(self.folder_name + 'optimization_kernel', 
#                          optimization_samples) 
            constant_kernel, corr_length = np.exp(self.gp.kernel_.theta)
            
            
            kernel_time = time.time() - start_time - qaoa_time - bayes_time
            step_time = time.time() - start_time
            
            solution = self.qaoa.solution_energy
            gs_en = self.qaoa.gs_en
            new_data = (
                        (i+self.nwarmup +1,
                         next_point,  
                         y_next_point, 
                         solution, 
                         1 - y_next_point/solution, 
                         qaoa_results['energy_exact'],
                         gs_en,
                         1 - qaoa_results['energy_exact']/gs_en,
                         energy_best,
                         qaoa_results['variance_sampled'], 
                         qaoa_results['variance_exact'],
                         qaoa_results['fidelity_sampled'],
                         qaoa_results['fidelity_exact'], 
                         fidelity_best,
                         qaoa_results['solution_ratio'], 
                         solution_ratio_best,
                         corr_length, 
                         constant_kernel, 
                         std_pop_energy, 
                         avg_sqr_distances, 
                         n_it, 
                         bayes_time, 
                         qaoa_time, 
                         kernel_time, 
                         step_time,
                         qaoa_results['doppler_detune'],
                         qaoa_results['actual_pulse_parameters'],
                         qaoa_results['bad_atoms'],
                         qaoa_results['final_state']
                         )
                        )
            
            with open(self.info_file_name, 'a') as f:
                f.write(f'iteration: {i +1}/{self.nbayes}  {next_point}'
                        f' (E - E_0)/E_0: {1 - y_next_point/self.qaoa.solution_energy}'
                        f' en: {y_next_point}'
                         ' fid: {}\n'.format(qaoa_results['fidelity_sampled']))
                        
            print(f'iteration: {i +1}/{self.nbayes}  {next_point}'
                    f' (E - E_0)/E_0: {1 - y_next_point/self.qaoa.solution_energy}'
                    ' en: {}, fid: {}, ratio: {}'.format(y_next_point, qaoa_results['fidelity_sampled'], qaoa_results['solution_ratio'])
                    )
                    
            
                    
            self.data_.append(new_data)
            simplified_data.append([i, 
                               next_point, 
                               1 - y_next_point/self.qaoa.solution_energy,
                               qaoa_results['fidelity_sampled'],
                               qaoa_results['solution_ratio']])
            df_simplified = pd.DataFrame(data = simplified_data, columns = ['ITER', 'POINT', '(E - E0)/E0', 'FID', 'ratio'])
            df_simplified.to_csv(self.folder_name + self.file_name + '_simplified.dat')
            #df = pd.DataFrame(data = self.data_, columns = self.data_names)
            #df.to_pickle(
            #            self.folder_name + self.file_name,
            #            )
            
        best_x, best_y, where = self.gp.get_best_point()
        self.data_.append(self.data_[where])
        simplified_data.append(simplified_data[where])
        df_simplified = pd.DataFrame(data = simplified_data, columns = ['ITER', 'POINT', '(E - E0)/E0', 'FID', 'ratio'])
        df_simplified.to_csv(self.folder_name + self.file_name + '_simplified.dat')
        #df = pd.DataFrame(data = self.data_, columns = self.data_names)
        #df.to_pickle(self.folder_name + self.file_name)
        print('Best point: ' , self.data_[where])
