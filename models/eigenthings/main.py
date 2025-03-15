import torch
import random
import argparse
import os
import json
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models import resnet20, cnn, lstm, ShakespeareDS
from hessian_eigenspectrum import *

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    
    print("creating dataset")
    ds, num_classes = get_datasets(args.dataset, args.test_set)
    dl = DataLoader(ds, batch_size=args.batch_size)
    print("dataset created")
    
    print("loading model")
    model = get_model(args.model, num_classes)
    print("model loaded")
    
    criterion = nn.CrossEntropyLoss()
    
    print("")
    with open(args.file, mode="a+") as f:
        for ckpt in args.ckpts:
            model.load_state_dict(torch.load(ckpt, map_location=args.device)[args.key])
            
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            
            if args.dataset == "shakespeare":
                with torch.backends.cudnn.flags(enabled=False):
                    print("computing the eigenvalue")
                    eigenvals, _ = compute_hessian_eigenthings(model, dl, criterion, args.n, mode="power_iter", power_iter_steps=args.max_steps)
            else:
                eigenvals, _ = compute_hessian_eigenthings(model, dl, criterion, args.n, mode="power_iter")
        
            f.write(f"{ckpt}: {eigenvals}")    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpts',
                        nargs="*",
                        type=str)
    parser.add_argument('-seed',
                        type=int,
                        default=0)
    parser.add_argument('-dataset',
                        required=True,
                        type=str)
    parser.add_argument('-device',
                        type=str,
                        default="cuda:0")
    parser.add_argument('-key',
                        type=str,
                        default="model_state_dict")
    parser.add_argument('--test-set',
                        action="store_true")
    parser.add_argument('--batch-size',
                        type=int,
                        default=64)
    parser.add_argument('-n',
                        type=int,
                        default=1)
    parser.add_argument('-file',
                        type=str,
                        default="results.txt")
    parser.add_argument('-model',
                        required=True,
                        type=str)
    parser.add_argument("--max-steps",
                        type=int,
                        default=20)
    return parser.parse_args()

def get_datasets(ds_name, test):
    if ds_name == 'cifar100':
        if not test:
            ds = datasets.CIFAR100("../data", train=True, download=True, transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                        ]))
        else:
            ds = datasets.CIFAR100("../data", train=False, download=True, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                        ]))
        num_classes = 100
    elif ds_name == 'cifar10':
        if not test:
            ds = datasets.CIFAR10("../data", train=True, download=True, transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
        else:
            ds = datasets.CIFAR10("../data", train=False, download=True, transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
        num_classes = 10
        num_classes = 80
                
    return ds, num_classes

def get_model(model_name, num_classes):
    if model_name == 'cnn':
        return cnn(num_classes)
    
if __name__ == "__main__":
    main()
