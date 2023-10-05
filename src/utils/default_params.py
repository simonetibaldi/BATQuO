
# Set global parameters
s_eigenvalues = [0, 1]
DEFAULT_PARAMS = {"penalty": 5,
                  "shots": 64,
                  "num_grid": 4,
                  "seed" : 22, 
                  "initial_length_scale" : 1,
                  "length_scale_bounds" : (0.01, 100),
                  "initial_sigma":1,
                  "constant_bounds":(0.01, 100),
                  "nu" : 2.5,
                  "max_iter_lfbgs": 50000,
                  "normalize_y": False,
                  "gtol": 1e-6,
                  "optimizer_kernel":'fmin_l_bfgs_b', #'fmin_l_bfgs_b' or #monte_carlo', 
                  "diff_evol_func": None, # or 'mc',
                  "n_restart_kernel_optimizer":9,
                  "distance_conv_tol": 0.01,
                  "angle_bounds": [[100, 1000], [100, 1000]]
                  }
                  
Q_DEVICE_PARAMS = {'type_of_lattice': 'triangular',
                   'thermal_motion': 85, #nm
                   'doppler_shift': 0.47, #MHz
                   'intensity_fluctuation':0.03, 
                   'laser_waist': 148, #micrometers
                   'rising_time': 50, #ns
                   'eta': 0.005, #state prep error for every qubit
                   'epsilon': 0.03, #false positive
                   'epsilon_prime': 0.08, #false negative
                   'temperature': 30, #microKelvin
                   'coherence_time': 5000, #ns
                   'omega_over_2pi': 1, #see notes/info.pdf for this value
                   'omega_off_over_2pi': 0
}