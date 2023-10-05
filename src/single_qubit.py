import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pulser import Pulse, Sequence, Register, Simulation
from pulser.devices import Chadoq2
Chadoq2.change_rydberg_level(60)
from pulser.simulation import SimConfig

from itertools import product
from utils.default_params import *
import random
from qutip import *
from scipy.stats import qmc
import pandas as pd

         
'''                
##### single qubit ham #####

H = omega X + delta Z
'''
omega = np.pi
delta = np.pi
pulse_length = 10000
dephasing_noise = False

dephase_noise = 0.1
qubits_dict = dict(enumerate([[0,0]]))     
reg = Register(qubits_dict)          
                
seq = Sequence(reg, Chadoq2)
seq.declare_channel('ch0','rydberg_global')
        
pulse = Pulse.ConstantPulse(pulse_length, omega, delta, 0)
            
seq.add(pulse, 'ch0')
            
seq.measure('ground-rydberg')

if dephasing_noise:
    noise_config = SimConfig(noise='dephasing', 
                            dephasing_prob = 1)
    print(f'dephasing prob: {noise_config.dephasing_prob}\n')
else:
    noise_config = None  


sim = Simulation(sequence = seq, 
                config = noise_config,
                sampling_rate = 1,
                evaluation_times = 0.1
                )
print('Simulation hamiltonian')
print(sim._hamiltonian)
print('Collpase op:')
print(sim._collapse_ops)
results_noiseless = sim.run()
#print('last state:\n',final_state)

times = sim._eval_times_array
r = [1, 0]  # |r>
g = [0, 1]  # |g>
occup_rr = [np.outer(r, np.conj(r))]  # |r><r|
occup_gg = [np.outer(g, np.conj(g))]

print('\nmeasuring:\n ', occup_rr)
print(occup_gg)

occupation_rr = np.array(results_noiseless.expect(occup_rr)).squeeze()
occupation_gg = np.array(results_noiseless.expect(occup_gg)).squeeze()
occupation_rr_incoherent = []
occupation_gg_incoherent = []

# first_el = results_noiseless.states[0]
# 
# for t in times:
# 
#     rho_noiseless = el * el.dag()
#     rho_final = ((1 - dephase_noise) * rho_noiseless + dephase_noise/3 * sigmax() * rho_noiseless *  sigmax()
#                                                     + dephase_noise/3 * sigmay() * rho_noiseless *  sigmay()
#                                                     + dephase_noise/3 * sigmaz() * rho_noiseless *  sigmaz()  )                                                  
#     
#     #print(rho_noiseless)
#     #print(rho_final)
#     occup_rr = (qeye(2) + sigmaz()) / 2
#     occup_gg = (qeye(2) - sigmaz()) / 2
#     occupation_rr_incoherent.append(expect(occup_rr, rho_final))
#     occupation_gg_incoherent.append(expect(occup_gg, rho_final))
# 
# print(occupation_rr_incoherent)


########################


dephasing_noise = True


qubits_dict = dict(enumerate([[0,0]]))     
reg = Register(qubits_dict)          
                
seq = Sequence(reg, Chadoq2)
seq.declare_channel('ch0','rydberg_global')
        
pulse = Pulse.ConstantPulse(pulse_length, omega, delta, 0)
            
seq.add(pulse, 'ch0')
            
seq.measure('ground-rydberg')

if dephasing_noise:
    noise_config = SimConfig(noise='dephasing', 
                            dephasing_prob = dephase_noise)
    print(f'dephasing prob: {noise_config.dephasing_prob}\n')
else:
    noise_config = None  


sim = Simulation(sequence = seq, 
                config = noise_config,
                sampling_rate = 1,
                evaluation_times = 0.1
                )
print('Simulation hamiltonian')
print(sim._hamiltonian)
print('Collpase op:')
print(sim._collapse_ops)
results = sim.run()
print('last state:\n',results.states[-1])
times = sim._eval_times_array
r = [1, 0]  # |r>
g = [0, 1]  # |g>
occup_rr = [np.outer(r, np.conj(r))]  # |r><r|
occup_gg = [np.outer(g, np.conj(g))]


psi_0 = basis(2,1)
c_ops = sim._collapse_ops
occup_rr = Qobj(np.outer(r, np.conj(r)), dims = [[2], [2]])
print(sim._hamiltonian.ops[0].get_qobj(1))
res = mcsolve(H = sim._hamiltonian.ops,
              psi0 = psi_0,
              tlist = times,
              c_ops = c_ops
              )
print(res)
exit()

print('\nmeasuring:\n ', occup_rr)
print(occup_gg)

occupation_rr = np.array(results.expect(occup_rr)).squeeze()
occupation_gg = np.array(results.expect(occup_gg)).squeeze()
                
                
fig = plt.figure()
plt.plot(times, occupation_gg_incoherent, label = 'incoherent')   
plt.plot(times, occupation_gg, label = 'ground')     
plt.legend()
plt.xlabel('time $(\mu s)$')
plt.ylabel('occupation')
plt.title(f'H = {omega:.2f}*X +{delta:.2f}*Z')
plt.show()  



                
