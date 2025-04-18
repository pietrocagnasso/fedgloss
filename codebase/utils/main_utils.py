import copy
import matplotlib.pyplot as plt
import os
import torch
from utils.args import SA_MINIMIZERS


def create_paths(args, current_time, alpha=None, resume=False):
    """ Create paths for checkpoints, plots, analysis results and experiment results. """
    model = args.model
    dataset = args.dataset

    # Create file for storing results
    res_path = os.path.join(args.dir, "results", f"{model}_{dataset}_{alpha}_{current_time}")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    ckpt_path = os.path.join(args.dir, "results", f"{model}_{dataset}_{alpha}_{current_time}", 'checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    run_info = 'K' + str(args.C_t) + '_N' + str(args.T) + '_clr' + str(args.lr) + '_' +\
                args.algorithm
    if args.server_opt is not None and args.server_lr is not None:
        run_info += '_' + args.server_opt + '_slr' + str(args.server_lr)
    run_info += '_' + current_time

    ckpt_name = None
    if alpha is not None:
        file = os.path.join(res_path, 'logs.txt')
        if not resume:
            ckpt_name = os.path.join(ckpt_path, '{}.ckpt'.format(str(alpha) + run_info))
    else:
        file = os.path.join(res_path, 'logs.txt')
        if not resume:
            ckpt_name = os.path.join(ckpt_path, '{}.ckpt'.format(run_info))

    results = open(os.path.join(res_path, f"trends.csv"), "w")
    eigs = open(os.path.join(res_path, f"eigs.txt"), "w")
    logger = open(file, "w")
    with open(os.path.join(res_path, f"params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(" ".join([k, ":", str(v), "\n"]))

    return ckpt_path, res_path, ckpt_name, results, eigs, logger

def get_alpha(dataset):
    data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    train_file = os.listdir(data_dir)
    if not train_file:
        print("Expected training file. Not found.")
        exit(-1)
    alpha = train_file[0].split('train_')[1][:-5]
    return alpha

def check_init_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            print("The path", path, "does not exist. Please specify a valid one.")
            exit(-1)

def define_server_params(args, client_model, server_name, opt_ckpt, tot_clients):
    if server_name == 'fedavg':
        server_params = {'client_model': client_model}
    elif server_name == 'fedopt':
        server_params = {'client_model': client_model, 'server_opt': args.server_opt,
                         'server_lr': args.server_lr, 'momentum': args.server_momentum,
                         'opt_ckpt': opt_ckpt}
    elif server_name == "fedgloss":
        server_params = {
            'client_model': client_model, 'server_opt': args.server_opt, 'server_lr': args.server_lr,
            'momentum': args.server_momentum, 'opt_ckpt': opt_ckpt, "rho_l": args.rho_l, "rho0": args.rho0,
            "T_s": args.T_s, "T": args.T, "rho": args.rho, "beta": args.beta, "tot_clients": tot_clients
        }
    else:
        raise NotImplementedError
    
    return server_params

def define_client_params(client_name, args, tot_clients):
    client_params = {'seed': args.seed, 'lr': args.lr, 'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
                     'num_workers': args.num_workers, 'momentum': args.momentum, "dataset": args.dataset}
    
    if client_name in SA_MINIMIZERS:
        client_params['rho_l'] = args.rho_l
        client_params['eta'] = args.eta
    if client_name == "fedgloss":
        client_params["beta"] = args.beta
    return client_params

def schedule_cycling_lr(round, c, lr1, lr2):
    t = 1 / c * (round % c + 1)
    lr = (1 - t) * lr1 + t * lr2
    return lr

def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn

def plot_metrics(accuracy, loss, n_rounds, figname, figpath, title, prefix='val_'):
    name = os.path.join(figpath, figname)

    plt.plot(n_rounds, loss, '-b', label=prefix + 'loss')
    plt.plot(n_rounds, accuracy, '-r', label=prefix + 'accuracy')

    plt.xlabel("# rounds")
    plt.ylabel("Average accuracy")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(name + '.png')  # should before show method
    plt.close()

def get_plots_name(args, current_time, alpha=None):
    if alpha is None:
        img_name_val = 'val_N' + str(args.T) + '_K' + str(args.C_t) + '_lr' + str(
            args.lr) + current_time
        img_name_test = 'test_N' + str(args.T) + '_K' + str(args.C_t) + '_lr' + str(
            args.lr) + current_time
    else:
        img_name_val = str(alpha) + '_val_N' + str(args.T) + '_K' + str(
            args.C_t) + '_lr' + str(args.lr) + current_time
        img_name_test = str(alpha) + '_test_N' + str(args.T) + '_K' + str(
            args.C_t) + '_lr' + str(args.lr) + current_time
    return img_name_val, img_name_test
