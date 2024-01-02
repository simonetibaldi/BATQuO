import argparse


def parse_command_line():
    parser = argparse.ArgumentParser(description="Run QAOA on Qiskit")

    parser.add_argument('--seed',
                        type=int,
                        default=12,
                        help="Seed for the random number generators"
                        )

    parser.add_argument('--p',
                        type=int,
                        default=1,
                        help="QAOA level number"
                        )

    parser.add_argument('--num_nodes',
                        type=int,
                        default=6,
                        help="Number of nodes in the graph"
                        )

    parser.add_argument('--average_connectivity',
                        type=float,
                        default=0.4,
                        help="Probability of selecting an"
                             "edge in the random graph"
                        )

    parser.add_argument('--nbayes',
                        type=int,
                        default=100,
                        help="Number of bayesian optimization steps"
                        )

    parser.add_argument('--fraction_warmup',
                        type=float,
                        default=0.1,
                        help="Fraction of warmup points"
                        )

    parser.add_argument('--nwarmup',
                        type=int,
                        default=10,
                        help="Number of warmup points"
                        )

    parser.add_argument('--i_trial',
                        type=int,
                        default=1,
                        help="Trial number"
                        )

    parser.add_argument('--trials',
                        type=int,
                        default=5,
                        help="Number of different run with the same graph"
                        )

    parser.add_argument('--dir_name',
                        type=str,
                        default="./",
                        help="Directory for saving data"
                        )
                    
    parser.add_argument('--quantum_noise',
                        type=str,
                        default=None,
                        help="Noise to decide between: None, SPAM, dephasing, doppler, all"
                        )
    parser.add_argument('--n_qubits',
                        type=int,
                        default=6,
                        help="number of qubits in the graph, 6 and 11 are diamond/butterfly"
                        )
                        
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help="level of verbose \n 0: print nothing \n 1: main training info \n 2: also saves info_file.txt"
                        )
                        
    parser.add_argument('--kernel',
                        type=str,
                        default='matern',
                        help="kernel type: matern or RBF"
                        )
    
    parser.add_argument('--shots',
                        type=int,
                        default=128,
                        help="Number of shots"
                        )
    
    parser.add_argument('--lattice_spacing',
                        type=float,
                        default=5,
                        help="spacing between qubits on the lattice (min 4)"
                        )
                        
    parser.add_argument('--discard_percentage',
                        type=float,
                        default=0,
                        help="percentage of higher energy measured states to be discarded"
                        )
                        
    return parser.parse_args()
