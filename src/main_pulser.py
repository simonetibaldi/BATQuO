import time
import random
import datetime
import sys
import numpy as np
from utils.parameters import parse_command_line
from utils.bayesian_optimization import *
from utils.default_params import *

np.set_printoptions(precision = 4, suppress = True)
np.set_printoptions(threshold=sys.maxsize)


###### TRAIN PARAMETERS ##################

args = parse_command_line()

seed = args.seed
fraction_warmup = args.fraction_warmup
depth = args.p
num_nodes = args.num_nodes
nwarmup = args.nwarmup
nbayes = args.nbayes
average_connectivity = args.average_connectivity
quantum_noise =  args.quantum_noise
type_of_graph = args.type_of_graph
lattice_spacing = args.lattice_spacing
verbose_ = args.verbose
kernel_choice = args.kernel
shots = args.shots

np.random.seed(seed)
random.seed(seed)

####### CREATE BAYES OPT INSTANCE ########
bo = Bayesian_optimization(depth,
                           type_of_graph,
                           lattice_spacing,
                           quantum_noise,
                           nwarmup,
                           nbayes,
                           kernel_choice,
                           shots,
                           seed,
                           verbose_
                           )
bo.print_info()        
bo.init_training(nwarmup)
bo.run_optimization()