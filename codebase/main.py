import importlib
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import torch
import atexit
import torchvision.transforms as transforms

from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict

import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from utils.args import parse_args, check_args
from utils.cutout import Cutout
from utils.main_utils import *
from utils.model_utils import read_data
from eigenthings.hessian_eigenspectrum import compute_hessian_eigenthings
from torchvision.datasets import CIFAR10, CIFAR100

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def close_open_files(files):
    for f in files:
        f.close()

def main():
    # read cli args
    args = parse_args()
    check_args(args)

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(mode=False)

    alpha = args.dir_alpha
    if alpha is not None:
        alpha = 'alpha={:.2f}'.format(alpha)
        print("Alpha:", alpha)

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    print("Using device:", torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')

    # Obtain the path to client's model (e.g. cifar10/cnn.py), client class and servers class
    model_path = '%s/%s.py' % (args.dataset, args.model)
    dataset_path = '%s/%s.py' % (args.dataset, 'dataloader')
    server_path = 'servers/%s.py' % (args.algorithm + '_server')
    client_path = 'clients/%s.py' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')
    check_init_paths([model_path, dataset_path, server_path, client_path])

    model_path = '%s.%s' % (args.dataset, args.model)
    dataset_path = '%s.%s' % (args.dataset, 'dataloader')
    server_path = 'servers.%s' % (args.algorithm + '_server')
    client_path = 'clients.%s' % (args.client_algorithm + '_client' if args.client_algorithm is not None else 'client')

    # Load model and dataset
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    dataset = importlib.import_module(dataset_path)
    ClientDataset = getattr(dataset, 'ClientDataset')

    # Load client and server
    print("Running experiment with server", server_path, "and client", client_path)
    Client, Server = get_client_and_server(server_path, client_path)
    print("Verify client and server:", Client, Server)

    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.T if args.T != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.C_t if args.C_t != -1 else tup[2]

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with servers model
    client_model = ClientModel(*model_params, device)

    client_model = client_model.to(device)
    
    # Create the server
    server_params = define_server_params(args, client_model, args.algorithm,
                                         None,
                                         tot_clients=100)
    server = Server(**server_params)

    # Create and set up clients
    train_clients, test_clients = setup_clients(args, client_model, Client, ClientDataset, device)
    train_client_ids, train_client_num_samples = server.get_clients_info(train_clients)
    test_client_ids, test_client_num_samples = server.get_clients_info(test_clients)
    if set(train_client_ids) == set(test_client_ids):
        print('Clients in Total: %d' % len(train_clients))
    else:
        print(f'Clients in Total: {len(train_clients)} training clients and {len(test_clients)} test clients')
    
    server.set_num_clients(len(train_clients))
    
    start_round = 0
    print("Start round:", start_round)

    # Initial status
    print('--- Random Initialization ---')

    start_time = datetime.now()
    current_time = start_time.strftime("%Y%m%dT%H:%M:%S")

    ckpt_path, res_path, ckpt_name, results_file, eigs_file, logger_file = create_paths(args, current_time, alpha=alpha)
    ckpt_name = current_time + '.ckpt'

    last_accuracies = []
    avg_model = OrderedDict()
    for k, v in server.model.items():
        avg_model[k] = torch.zeros_like(v)

    atexit.register(close_open_files, [results_file, eigs_file, logger_file])

    print_stats(start_round, server, train_clients, train_client_num_samples, test_clients, test_client_num_samples,
                args, logger_file)

    results_file.write("round,accuracy,loss,model_norm,pseudograd_norm\n")
    # Start training
    for i in range(start_round, num_rounds):
        
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
        logger_file.write('--- Round %d of %d: Training %d Clients ---\n' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)
        logger_file.write(f'Selected clients: {c_ids}' + '\n')

        ##### Simulate servers model training on selected clients' data #####
        sys_metrics = server.train_model(num_epochs=args.E, batch_size=args.batch_size,
                                         minibatch=args.minibatch)

        ##### Update server model #####
        print("--- Updating central model ---")
        server.update_model()

        ##### Test model #####
        # eveluation is performed every <eval_every> round and every round in the last 100
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds or (i+1) > num_rounds - 100:
            _, test_metrics = print_stats(i + 1, server, train_clients, train_client_num_samples, test_clients, test_client_num_samples,
                                                args, logger_file)
            if (i+1) > num_rounds - 100:
                last_accuracies.append(test_metrics[0])
                for k in avg_model:
                    avg_model[k] += server.model[k] / 100

            ### Gradients information ###
            model_grad_norm = server.get_model_grad()
            model_params_norm = server.get_model_params_norm()

            results_file.write(f"{i+1},{test_metrics[0]},{test_metrics[1]},{model_params_norm},{model_grad_norm}" + "\n")

            # Save round global model checkpoint
            where_saved = server.save_model(i + 1, os.path.join(ckpt_path, ckpt_name))
            print('Checkpoint saved in path: %s' % where_saved)

    ## FINAL ANALYSIS ##
    where_saved = server.save_model(num_rounds, os.path.join(ckpt_path, 'round:' + str(num_rounds) + '_' + current_time + '.ckpt'))
    print('Checkpoint saved in path: %s' % where_saved)

    if last_accuracies:
        avg_acc = sum(last_accuracies) / len(last_accuracies)
        print("Last {:d} rounds accuracy: {:.3f}".format(len(last_accuracies), avg_acc))
    
    if not args.no_eigs:
        # compute the Hessian eigenvalues
        server.client_model.load_state_dict(avg_model)
        compute_and_log_eigs(args, server.client_model, eigs_file)

    # Save results
    logger_file.close()
    results_file.close()
    eigs_file.close()

    if args.plots:
        # generate accuracy and loss plots for the run
        trends_df = pd.read_csv(os.path.join(".", "results", f"{current_time}", f"trends.csv"))
        trends_df.plot(x="round", y="accuracy", color="tab:blue", title="Accuracy trend")
        plt.xlabel("Round")
        plt.ylabel("Accuracy [%]")
        plt.savefig(os.path.join(res_path, f"accuracy.pdf"), bbox_inches="tight", pad_inches=0.1)
        plt.close()
        trends_df.plot(x="round", y="loss", color="tab:orange", title="Loss trend")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(res_path, f"loss.pdf"), bbox_inches="tight", pad_inches=0.1)
        plt.close()

    print("File saved in path: %s" % res_path)


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, train_data, test_data, model, args, ClientDataset, Client, run=None, device=None, tot_clients=None):
    clients = []
    client_params = define_client_params(args.client_algorithm, args, tot_clients)
    client_params['model'] = model
    client_params['run'] = run
    client_params['device'] = device
    for u in tqdm(users):
        c_traindata = ClientDataset(train_data[u], train=True, loading=args.where_loading, cutout=Cutout if args.cutout else None, device=args.device)
        c_testdata = ClientDataset(test_data[u], train=False, loading=args.where_loading, cutout=None, device=args.device)
        client_params['client_id'] = u
        client_params['train_data'] = c_traindata
        client_params['eval_data'] = c_testdata
        clients.append(Client(**client_params))
    return clients


def setup_clients(args, model, Client, ClientDataset, device=None, run=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')

    train_users, train_groups, test_users, test_groups, train_data, test_data = read_data(train_data_dir, test_data_dir, args.dir_alpha)

    tot_clients = len(train_users)

    train_clients = create_clients(train_users, train_data, test_data, model, args, ClientDataset, Client, run, device, tot_clients)
    test_clients = create_clients(test_users, train_data, test_data, model, args, ClientDataset, Client, run, device, tot_clients)

    return train_clients, test_clients

def get_client_and_server(server_path, client_path):
    mod = importlib.import_module(server_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    server_name = server_path.split('.')[1].split('_server')[0]
    server_name = list(map(lambda x: x[0], filter(lambda x: 'Server' in x[0] and server_name in x[0].lower(), cls)))[0]
    Server = getattr(mod, server_name)
    mod = importlib.import_module(client_path)
    cls = inspect.getmembers(mod, inspect.isclass)
    client_name = max(list(map(lambda x: x[0], filter(lambda x: 'Client' in x[0], cls))), key=len)
    Client = getattr(mod, client_name)
    return Client, Server

def print_stats(num_round, server, train_clients, train_num_samples, test_clients, test_num_samples, args, fp):
    train_stat_metrics = server.test_model(train_clients, args.batch_size, set_to_use='train')
    val_metrics = print_metrics(train_stat_metrics, train_num_samples, fp, prefix='train_')

    test_stat_metrics = server.test_model(test_clients, args.batch_size, set_to_use='test' )
    test_metrics = print_metrics(test_stat_metrics, test_num_samples, fp, prefix='{}_'.format('test'))

    return val_metrics, test_metrics

def print_metrics(metrics, weights, fp, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    metrics_values = []
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        fp.write('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g\n' \
                 % (prefix + metric,
                    np.average(ordered_metric, weights=ordered_weights),
                    np.percentile(ordered_metric, 10),
                    np.percentile(ordered_metric, 50),
                    np.percentile(ordered_metric, 90)))
        # fp.write("Clients losses:", ordered_metric)
        metrics_values.append(np.average(ordered_metric, weights=ordered_weights))
    return metrics_values

def compute_and_log_eigs(args, model, eigs_file):
    ds = get_ds(args)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    
    eigenvals, _ = compute_hessian_eigenthings(model, dl, criterion, 5, mode="power_iter")
    
    for i, eig in enumerate(eigenvals):
        eigs_file.write(f"lambda_{i+1} = {eig}" + "\n")
    eigs_file.write(f"lambda_1 / lambda_5 = {eigenvals[0] / eigenvals[4]}")

def get_ds(args):
    ds_name = args.dataset
    if ds_name == 'cifar100':
        ds = CIFAR100(os.path.join("data/cifar100"), train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2616))
                ]))
    elif ds_name == 'cifar10':
        ds = CIFAR10(os.path.join("data/cifar10"), train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ]))
    
    return ds

if __name__ == '__main__':
    main()
