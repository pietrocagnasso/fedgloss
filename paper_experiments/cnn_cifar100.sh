#!/bin/bash

cd ../codebase || echo "Failed to change directory to ../codebase"

# cifar100 alpha=0
python main.py --dataset cifar100 --dir-alpha 0 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 0
python main.py --dataset cifar100 --dir-alpha 0 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 42
python main.py --dataset cifar100 --dir-alpha 0 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 200299

# cifar100 alpha=0.5
python main.py --dataset cifar100 --dir-alpha 0.5 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 0
python main.py --dataset cifar100 --dir-alpha 0.5 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 42
python main.py --dataset cifar100 --dir-alpha 0.5 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 200299

# cifar100 alpha=0
python main.py --dataset cifar100 --dir-alpha 1000 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 0
python main.py --dataset cifar100 --dir-alpha 1000 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 42
python main.py --dataset cifar100 --dir-alpha 1000 --where-loading init --model cnn -T 20000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.2 --eta 0 --beta 100 --rho 0.01 --T-s 15000 --plots --seed 200299
