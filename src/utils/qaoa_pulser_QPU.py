import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import random
import pandas as pd

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
from pulser.backend import EmulatorConfig

from pulser_pasqal import PasqalCloud



#connection = PasqalCloud(
#   username=username,  # Your username or email address for the Pasqal Cloud Platform
#   project_id=project_id,  # The ID of the project associated to your account
#   password=password,  # The password for your Pasqal Cloud Platform account
#)

#QPU_device = connection.fetch_available_devices()['FRESNEL']
device_used = AnalogDevice

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
        self.Fresnel = AnalogDevice
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
        self.G, self.qubits_dict = self.generate_graph(type_of_graph)
        self.solution, self.solution_energy = self.classical_solution()
        self.Nqubit = len(self.G)
        
             
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
        
        self.reg = trap_layout.define_register(*self.trap_list)
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
        
        hypercube_sampler = qmc.LatinHypercube(d=self.depth*2, seed = self.seed)
        X =  hypercube_sampler.random(N_points)
        l_bounds = self.angles_bounds[:,0]
        u_bounds = self.angles_bounds[:,1]
        X = qmc.scale(X, l_bounds, u_bounds).astype(int)
        X = X.tolist()
        for x in X:
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
        seq.measure('ground-rydberg')
        
        bknd = EmuFreeBackend(seq, connection=connection,)
        #bknd = QPUBackend(seq, connection=connection,)
        
        return bknd
        
    def quantum_loop(self, param):
        ''' Run the quantum circuits. It creates the circuit, add noise if 
        needed.
        '''
        
        bknd = self.create_quantum_circuit(param)
        
        #self.doppler_detune = sim._doppler_detune
        #self.noisy_pulse_parameters = sim.samples
                    
    
        results = bknd.run(job_params=[{"runs":self.shots,"variables":{}}])
        
        count_dict = {x:s for x,s in results[0].bitstring_counts.items()} 
    
        #self.bad_atoms = sim._bad_atoms
        
        if self.discard_percentage > 0 and len(count_dict)>2:
            ### order the count_dict (prolly already ordered by Pulser)
            count_dict = dict(sorted(count_dict.items(), 
                                key=lambda item: item[1], 
                                reverse=True))
            shots_to_erase = int(self.discard_percentage*self.shots)
            lowest_shots = list(count_dict.values())[-1]
            if lowest_shots < shots_to_erase:
                erased_shots = 0
                while erased_shots < shots_to_erase:
                    erased_shots += count_dict.popitem()[1]
        
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
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
