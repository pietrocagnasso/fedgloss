import argparse

DATASETS = ['cifar10', 'cifar100', 'shakespeare']
SERVER_ALGORITHMS = ['fedavg', 'fedopt', "fedgloss"]
SERVER_OPTS = ['sgd', 'adam', 'adagrad']
CLIENT_ALGORITHMS = ['sam', "fedgloss"]
SA_MINIMIZERS = ['sam', "fedgloss"]
SIM_TIMES = ['small', 'medium', 'large']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        type=str,
                        default=".")
    
    ## FEDERATED SETTING ##
    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True)
    parser.add_argument('--dir-alpha',
                        help="value for Dirichlet's alpha",
                        type=float,
                        default=None)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluation period in rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('-algorithm',
                        help='algorithm used for server aggregation;',
                        choices=SERVER_ALGORITHMS,
                        default=None)
    parser.add_argument('--client-algorithm',
                        help='algorithm used on the client-side for regularization',
                        choices=CLIENT_ALGORITHMS,
                        default=None)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting;',
                        type=int,
                        default=0)
    parser.add_argument("--no-eigs",
                        action="store_true",)

    ## SERVER OPTMIZER ##
    parser.add_argument('--server-opt',
                        help='server optimizer;',
                        choices=SERVER_OPTS,
                        required=False)
    parser.add_argument('--server-lr',
                        help='learning rate for server optimizers;',
                        type=float,
                        required=False)
    parser.add_argument('--server-momentum',
                        help='momentum for server optimizers;',
                        type=float,
                        default=0,
                        required=False)

    ## CLIENT TRAINING ##
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                                        help='None for FedAvg, else fraction;',
                                        type=float,
                                        default=None)
    epoch_capability_group.add_argument('--num-epochs',
                                        help='number of epochs when clients train on data;',
                                        type=int,
                                        default=1)
    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=-1,
                        required=False)
    parser.add_argument('--weight-decay',
                        help='weight decay for local optimizers;',
                        type=float,
                        default=0,
                        required=False)
    parser.add_argument('-momentum',
                        help='Client momentum for optimizer',
                        type=float,
                        default=0)
    parser.add_argument('-cutout',
                        help='apply cutout',
                        action='store_true',
                        default=False)

    ## GPU ##
    parser.add_argument('-device',
                        type=str,
                        default='cuda:0',
                        help="device on which the experiment is carried out;")

    ## DATALOADER ##
    parser.add_argument('--num-workers',
                        help='dataloader num workers',
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument('--where-loading',
                        help='location for loading data in ClientDataset',
                        type=str,
                        choices=['init', 'training_time'],
                        default='training_time',
                        required=False)

    ## FedSAM, FedASAM, SWA hyperparams
    parser.add_argument('-rho',
                        help='rho for sharpness-aware minimizers',
                        type=float,
                        default=None)
    parser.add_argument('-eta',
                        help='eta for sharpness-aware minimizers',
                        type=float,
                        default=None)
    parser.add_argument('--server-rho',
                        help="rho for the server-side optimization",
                        type=float,
                        default=None)
    parser.add_argument('-beta',
                        type=float,
                        help='lagrangian controller hyperparameter',
                        default=None)
    parser.add_argument('--rho-warmup-steps',
                        type=int,
                        default=0,
                        help="length of the warmup pahes on both server- and client-side optimization")
    parser.add_argument('-rho0',
                        type=float,
                        default=0.001,
                        help="starting value for rho used when warmup steps > 0")
    parser.add_argument('--sharpness-mom',
                        help="momentum for server-side sharpness term",
                        type=float,
                        default=0.)

    ## ANALYSIS OPTIONS ##
    parser.add_argument('--metrics-name',
                        help='name for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('--metrics-dir',
                        help='dir for metrics file;',
                        type=str,
                        default='metrics',
                        required=False)
    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')

    return parser.parse_args()

def check_args(args):
    if args.algorithm == "fedgloss":
        args.client_algorithm = "fedgloss"

    if (args.client_algorithm in SA_MINIMIZERS) and (args.rho is None or args.eta is None):
        print("Specificy values for rho, eta to run with a client-side sharpness-aware opt")
        exit(-1)
    
    if args.algorithm == "fedgloss" and (args.beta is None or args.server_rho is None):
        print("Specify beta and server_rho to run fedgloss")
        exit(-1)
