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
depth = args.p
nwarmup = args.nwarmup
nbayes = args.nbayes
type_of_graph = args.type_of_graph
lattice_spacing = args.lattice_spacing
verbose_ = args.verbose
kernel_choice = args.kernel
shots = args.shots
discard_percentage = args.discard_percentage
load_data = args.load_data

np.random.seed(seed)
random.seed(seed)

####### CREATE BAYES OPT INSTANCE ########
bo = Bayesian_optimization(depth,
                           type_of_graph,
                           lattice_spacing,
                           nwarmup,
                           nbayes,
                           kernel_choice,
                           shots,
                           discard_percentage,
                           seed,
                           verbose_
                           ) 
bo.save_info() 
if load_data==None:    
    bo.init_training()
bo.run_optimization(load_data)