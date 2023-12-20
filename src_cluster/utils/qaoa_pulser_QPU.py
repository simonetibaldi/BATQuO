import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import random
import pandas as pd
import time
import os
import pickle
import json
from collections import Counter

from pulser.sequence import Sequence
from pulser_simulation import Simulation, SimConfig
from pulser.devices import Device
from pulser.channels import Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import TriangularLatticeLayout
from pulser.devices import AnalogDevice

from pulser_simulation import QutipBackend
from pulser_pasqal import EmuFreeBackend, EmuTNBackend
from pulser import QPUBackend
from pulser.backend import EmulatorConfig,NoiseModel

from pulser_pasqal import PasqalCloud

username = "lucas.leclerc@pasqal.com"
project_id = "a210fe93-2276-4138-9d03-64e1824d1c4e"
password = "Luc92cloud!"


connection = PasqalCloud(
   username=username,  # Your username or email address for the Pasqal Cloud Platform
   project_id=project_id,  # The ID of the project associated to your account
   password=password,  # The password for your Pasqal Cloud Platform account
)

device_used = AnalogDevice
device_used = connection.fetch_available_devices()['FRESNEL']

from utils.default_params import *
from qutip import *
from scipy.stats import qmc

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class qaoa_pulser(object):

    def __init__(self, 
                 depth, 
                 angles_bounds,
                 type_of_graph, 
                 lattice_spacing, 
                 shots,
                 discard_percentage,
                 seed):
                
        self.seed = seed
        self.shots = shots
        self.Fresnel = device_used
        self.discard_percentage = discard_percentage
        self.C_6_over_h = self.Fresnel.interaction_coeff
        self.omega = Q_DEVICE_PARAMS['omega_over_2pi'] * 2 * np.pi 
        self.omega_off = Q_DEVICE_PARAMS['omega_off_over_2pi'] * 2 * np.pi
        self.delta = -np.pi
        
        self.rydberg_radius = self.Fresnel.rydberg_blockade_radius(self.omega)
        self.lattice_spacing = lattice_spacing
        self.U = [] # it is a list because two qubits in rydberg 
                    # interaction might be closer than others
        self.angles_bounds = angles_bounds
        self.depth = depth
        
        self.type_of_graph = type_of_graph
        self.G, self.qubits_dict = self.generate_graph(type_of_graph)
        self.Nqubit = len(self.G)
        if type_of_graph != "big":
            self.solution, self.solution_energy = self.classical_solution()
        else:
            with open('results_big_graph.json', 'rb') as f:
                d = json.load(f)
                en = list(d.values())[0]
                self.solution = d
                self.solution_energy = en
        
        
        
             
    def Omega_from_b(self, omega, delta_i, omega_r):
        
        return omega * 2 * delta_i / omega_r
        
    def lightshift_function(self, omega):
        delta_i = 700 * 2 * np.pi #MHz
        Omega_r = 30  * 2 * np.pi #MHz
        
        l_shift = (Omega_r**2 
                   - self.Omega_from_b(omega, delta_i, Omega_r)**2)/(4*delta_i)
                     
        return l_shift
        
    def delta_from_omega(self, omega_on, omega_off):
    
        return self.lightshift_function(omega_on) \
                - self.lightshift_function(omega_off)
        
    def generate_graph(self, type_of_graph): 
        '''
        Creates a networkx graph from the relative positions between qubits
        Parameters: positions of qubits in micrometers
        Returns: networkx graph G
        '''
        a = 5.
        N_max_atoms = 11
        trap_layout = list(device_used.calibrated_register_layouts.values())[0]
        if type_of_graph == 'diamond':
            pos_ =[
                  [0, 0], 
                  [a, 0], 
                  [3/2 * a, np.sqrt(3)/2 * a], 
                  [3/2 * a, -np.sqrt(3)/2 * a], 
                  [2 * a, 0], 
                  [3 * a, 0]
                  ]
            self.trap_list = [21, 30, 35, 34, 39, 48]
        elif type_of_graph == 'butterfly':
            pos_ = [
                   [0, 0],
                   [0, a],
                   [0, a * 2],
                   [np.sqrt(3) * 1/2 * a, 1/2 * a],
                   [np.sqrt(3) * 1/2 * a, 3/2 * a],
                   [np.sqrt(3) * a, a],
                   [np.sqrt(3) * 3/2 * a, 1/2 * a],
                   [np.sqrt(3) * 3/2 * a, 3/2 * a],
                   [np.sqrt(3) * 2 * a, 0],
                   [np.sqrt(3) * 2 * a, a],
                   [np.sqrt(3) * 2 * a, a * 2]
            ]
            self.trap_list = [38, 29, 20, 34, 25, 30, 35, 26, 40, 31, 22]
        elif type_of_graph == 'thirteen':
            pos_ = [
                   [0, 0],
                   [0, a],
                   [0, a * 2],
                   [np.sqrt(3) * 1/2 * a, 1/2 * a],
                   [np.sqrt(3) * 1/2 * a, 3/2 * a],
                   [np.sqrt(3) * a, a],
                   [np.sqrt(3) * a, 2 * a],
                   [np.sqrt(3) * a, 3 * a],
                   [np.sqrt(3) * 3/2 * a, 1/2 * a],
                   [np.sqrt(3) * 3/2 * a, 3/2 * a],
                   [np.sqrt(3) * 2 * a, 0],
                   [np.sqrt(3) * 2 * a, a],
                   [np.sqrt(3) * 2 * a, a * 2]
            ]
            self.trap_list = [38, 29, 20, 34, 25, 30, 35, 26, 40, 31, 22,12,21,39,48]
            pos_ = trap_layout.coords[self.trap_list]
        elif type_of_graph == "big":
            self.trap_list = [27, 10, 28, 20, 29, 12, 33, 31, 30, 8, 17, 1, 13, 37, 9, 26, 34, 40, 15, 7, 36, 19, 44, 42, 0]
            pos_ = trap_layout.coords[self.trap_list]
        
        self.reg = trap_layout.define_register(*self.trap_list)
        self.reg.draw()
        G = nx.Graph()
        edges=[]
        distances = []
        for n in range(len(pos_)-1):
            for m in range(n+1, len(pos_)):
                pwd = ((pos_[m][0]-pos_[n][0])**2
                        +(pos_[m][1]-pos_[n][1])**2)**0.5
                distances.append(pwd)
                if pwd < self.rydberg_radius:
                    # Below rbr, vertices are connected
                    edges.append([n,m]) 
                    #And the interaction is given by C_6/(h*d^6)
                    self.U.append(self.C_6_over_h/(pwd**6)) 
        G.add_nodes_from(range(len(pos_)))
        G.add_edges_from(edges)
        
        return G, dict(enumerate(pos_))
        
    def classical_solution(self):
        '''
        Runs through all 2^n possible configurations and estimates the solution
        Returns: 
            d: dictionary with {[bitstring solution] : energy}
            en: energy of the (possibly degenerate) solution
        '''
        results = {}

        string_configurations = list(product(['0','1'], repeat=len(self.G)))

        for string_configuration in  string_configurations:
            single_string = "".join(string_configuration)
            results[single_string] = self.get_cost_string(string_configuration)
        
        d = dict((k, v) for k, v in results.items() if v == np.min(list(results.values())))
        en = list(d.values())[0]
        
        #sort the dictionary
        results = dict(sorted(results.items(), key=lambda item: item[1]))
        
        #counts the distribution of energies
        energies, counts = np.unique(list(results.values()), return_counts = True)
        df = pd.DataFrame(np.column_stack((energies, counts)), columns = ['energy', 'counts'])
        #print('\n####CLASSICAL SOLUTION######\n')
        #print('Lowest energy:', d)
        #print('First excited states:', {k: results[k] for k in list(results)[1:df.values[1,1]]})
        #print('Energy distribution')
        #print(df)
        
        
        return d, en
               
    def get_cost_string(self, string):
        'Receives a string of 0 and 1s and gives back its cost to the MIS hamiltonian'
        penalty = DEFAULT_PARAMS["penalty"]
        configuration = np.array(tuple(string),dtype=int)
        
        cost = 0
        delta = np.max([self.delta, -self.delta]) # to ensure the constant is negative
        cost = -sum(configuration)
        for i,edge in enumerate(self.G.edges):
            cost += penalty*(configuration[edge[0]]*configuration[edge[1]])
        
        return cost
        
    def get_cost_dict(self, counter):
        total_cost = 0
        for key in counter.keys():
            cost = self.get_cost_string(key)
            total_cost += cost * counter[key]
        return total_cost / sum(counter.values())
        
    def generate_random_points(self, N_points):
        ''' Generates N_points random points with the latin hypercube method
        
        Attributes:
        N_points: how many points to generate
        return_variance: bool, if to calculate the variance or not
        '''
        X , Y , data_train = [], [], []
        
       #  hypercube_sampler = qmc.LatinHypercube(d=self.depth*2, seed = self.seed)
#         X =  hypercube_sampler.random(N_points)
#         l_bounds = self.angles_bounds[:,0]
#         u_bounds = self.angles_bounds[:,1]
#         X = qmc.scale(X, l_bounds, u_bounds).astype(int)
#         X = X.tolist()
#         for x in X:
#             while (np.sum(x) > 3750):
#                  x = hypercube_sampler.random(1)
#                  x = qmc.scale(x, l_bounds, u_bounds).astype(int)[0]
#                  print('Sequence larger than 3 ns, proposing new training point')
        repeat = True
        accepted_X = []
        while repeat:
            hypercube_sampler = qmc.LatinHypercube(d=self.depth*2, seed = self.seed)
            X =  hypercube_sampler.random(N_points)
            l_bounds = self.angles_bounds[:,0]
            u_bounds = self.angles_bounds[:,1]
            X = qmc.scale(X, l_bounds, u_bounds).astype(int)
            X = X.tolist()
            for x in X:
                if sum(x)<(3900 - Q_DEVICE_PARAMS['first_pulse_duration']): 
                    accepted_X.append(x)
                    if len(accepted_X) >= N_points:
                        repeat = False
                        break
        for x in accepted_X:
            qaoa_results = self.apply_qaoa(x)
            Y.append(qaoa_results['energy_sampled'])
            data_train.append(qaoa_results)
            
        return X, Y, data_train

    def create_quantum_circuit(self, params):
        seq = Sequence(self.reg, self.Fresnel)
        seq.declare_channel('ch','rydberg_global')
        seq.enable_eom_mode("ch", amp_on=self.omega, detuning_on=0)
        
        ### FIRST pulse to rotate the qubits e^{-i pi/2 X}
        seq.add_eom_pulse("ch", duration=250, phase=0.0)
        
        gammas = params[::2] #Hc
        betas = params[1::2] #Mixing
        
        for i in range(self.depth):
            #Ensures params are multiples of 4 ns
            beta_i = int(betas[i]) - int(betas[i]) % 4
            gamma_i = int(gammas[i]) - int(gammas[i]) % 4
            
            seq.delay(gamma_i, "ch")
            seq.add_eom_pulse("ch", duration=beta_i, phase=0.0)
        seq.disable_eom_mode("ch")
        seq.measure('ground-rydberg')
        
        #bknd = QutipBackend(seq,EmulatorConfig(sampling_rate=0.1,with_modulation=True,noise_model=NoiseModel(noise_types=["SPAM"],samples_per_run=self.shots,runs=1,p_false_pos=3e-2,p_false_neg=7e-2)))
        bknd = EmuFreeBackend(seq, connection=connection,)
        #bknd = QPUBackend(seq, connection=connection,)
        
        
        return bknd
        
    def quantum_loop(self, param):
        ''' Run the quantum circuits. It creates the circuit, add noise if 
        needed.
        '''
        def truncate_randomly_bitstrings(bitstrings,n_shots):
            S = sum(bitstrings.values())
            list_samples = random.choices(list(bitstrings.keys()), weights=np.array(list(bitstrings.values()))/S, k=n_shots)
            return dict(Counter(list_samples))
        
        ### LOAD EXISTING X0
        folder_name = "p={}_".format(self.depth)+self.type_of_graph+"_shots_{}_seed_{}".format(self.shots,self.seed)
        if os.path.exists("output/"+folder_name+"_opt_history.json"):
            with open("output/"+folder_name+"_opt_history.json", 'rb') as fp:
                data = json.load(fp)
        else: 
            data = {"x0":[],"id":[],"bitstrings":[],"n_shots":[]}
            
        if param not in data["x0"]:
                
            bknd = self.create_quantum_circuit(param)
            
            results = bknd.run(job_params=[{"runs":int(self.shots/0.75),"variables":{}}])
            
            check = True            
            while check:
                try:
                    status = results.get_status().name
                    if status != "DONE":
                        time.sleep(2)
                    else:
                        check = False
                except AssertionError:
                    time.sleep(60)
                    print("Fresnel appears offline, retrying in 60 seconds")
                    results = bknd.run(job_params=[{"runs":int(self.shots/0.75),"variables":{}}])
                    check = True 
                    
    
            count_dict = {x:s for x,s in results[0].bitstring_counts.items()} 
            count_dict = truncate_randomly_bitstrings(count_dict,self.shots)
            
            #count_dict = dict(bknd.run().sample_final_state())
            #count_dict = {x:int(k) for x,k in count_dict.items()}

            
            data["x0"].append(param)
            data["id"].append(results._submission_id)
            data["bitstrings"].append(count_dict)
            data["n_shots"].append(sum(count_dict.values()))
            with open("output/"+folder_name+"_opt_history.json", 'w') as fp:
                json.dump(data, fp)
        ### SAVE BITSTRINGS
        
        else:
            ind = data["x0"].index(param)
            data["x0"].append(param)
            data["id"].append(data["id"][ind])
            count_dict = data["bitstrings"][ind]
            data["bitstrings"].append(count_dict)
            data["n_shots"].append(sum(count_dict.values()))
            with open("output/"+folder_name+"_opt_history.json", 'w') as fp:
                json.dump(data, fp)
            
        #self.bad_atoms = sim._bad_atoms
        if self.discard_percentage > 0 and len(count_dict)>2:
            ### order the count_dict (prolly already ordered by Pulser)
            count_dict = dict(sorted(count_dict.items(), 
                                key=lambda item: self.get_cost_string(item[0]), 
                                reverse=False))
            shots_to_erase = int(self.discard_percentage*self.shots)
            while 0 < shots_to_erase:
                if list(count_dict.values())[-1] < shots_to_erase:
                    shots_to_erase += - count_dict.popitem()[1]
                else:
                    count_dict[list(count_dict.keys())[-1]] += - shots_to_erase
                    shots_to_erase = 0
        
        #if self.discard_percentage > 0 and len(count_dict)>2:
            ### order the count_dict (prolly already ordered by Pulser)
        #    count_dict = dict(sorted(count_dict.items(), 
        #                        key=lambda item: item[1], 
        #                        reverse=True))
        #    shots_to_erase = int(self.discard_percentage*self.shots)
        #    lowest_shots = list(count_dict.values())[-1]
        #    if lowest_shots < shots_to_erase:
        #        erased_shots = 0
        #        while erased_shots < shots_to_erase:
        #            erased_shots += count_dict.popitem()[1]
        
        return count_dict
            
    def calculate_solution_ratio(self, C):
        '''Calculates the ratio of the probability of the solution state vs
        the probability of the second most likely state when the solution state 
        is the most likely.
        
        If the solution state does not have the highest probability it returns 
        zero
        '''
        
        sol_ratio = 0
        sorted_dict = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        if len(sorted_dict)>1 :
            first_key, second_key =  list(sorted_dict.keys())[:2]
            if (first_key in self.solution.keys()):
                sol_ratio = C[first_key]/C[second_key]
        else:
            first_key =  list(sorted_dict.keys())[0]
            if (first_key in self.solution.keys()):
                sol_ratio = -1    #if the only sampled value is the solution the ratio is infinte so we put -1
            
        return sol_ratio
        
    def calculate_sampled_energy_and_variance(self, sampled_state):
        sampled_energy = self.get_cost_dict(sampled_state)
        
        estimated_variance = 0
        for configuration in list(sampled_state.keys()):
            hamiltonian_i = self.get_cost_string(configuration) # energy of i-th 
                                                                # configuration
            estimated_variance += sampled_state[configuration] * (sampled_energy - hamiltonian_i)**2
        
        estimated_variance = estimated_variance / (self.shots - 1) # use unbiased variance estimator
        
        return sampled_energy, estimated_variance

    def calculate_fidelity_sampled(self, C):
        '''
        Fidelity sampled means how many times the solution(s) is measured
        '''
        fid = 0
        for sol_key in self.solution.keys():
            if sol_key in list(C.keys()):
                fid += C[sol_key]
        
        fid = fid/self.shots
        
        return fid
        
    def apply_qaoa(self, params):
        '''
        Runs qaoa with the specified parameters
        
        Parameters
        ----------
            params: set of angles, needs to be of len 2*depth, can be either int or float
        
        Returns
        -------
            results_dict: Dictionary with all the info of the qaoa final state:
                          sampled_state, 
                          sampled_energy, 
                          sampled_variance, 
                          exact_energy
                          exact_variance, 
                          sampled_fidelity, 
                          exact_fidelity, 
                          solution_ratio
                          states of the evolution     
        '''
        #Checks the number of parameters
        if len(params) != 2*self.depth:
            print('\nWARNING:\n'
                  f'Running qaoa with a number of params different from '
                  f'2*depth = {2*self.depth}, number of passed parameters is '
                  f'{len(params)}')
                  
        #quantum loop returns a dictionary of N_shots measured states and the evolution
        #of qutip state
        sampled_state= self.quantum_loop(params)
        sampled_energy, sampled_variance = self.calculate_sampled_energy_and_variance(sampled_state)
        
        results_dict = {}
        results_dict['sampled_state'] = sampled_state
        results_dict['energy_sampled'] = sampled_energy
        results_dict['variance_sampled'] = sampled_variance
        results_dict['fidelity_sampled'] = self.calculate_fidelity_sampled(sampled_state)
        results_dict['approximation_ratio'] = 1 - sampled_energy/self.solution_energy
        results_dict['solution_ratio'] = self.calculate_solution_ratio(sampled_state)

        return results_dict
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                