#!/usr/bin/env bash

pushd ../codebase

# cifar10 alpha=0
python main.py --dataset cifar10 --dir-alpha 0 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 0 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 0 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299

# cifar10 alpha=0.05
python main.py --dataset cifar10 --dir-alpha 0.05 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 0.05 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 0.05 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299

# cifar10 alpha=1
python main.py --dataset cifar10 --dir-alpha 1 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 1 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 1 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299

# cifar10 alpha=5
python main.py --dataset cifar10 --dir-alpha 5 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 5 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 5 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299

# cifar10 alpha=10
python main.py --dataset cifar10 --dir-alpha 10 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 10 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 10 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299

# cifar10 alpha=100
python main.py --dataset cifar10 --dir-alpha 100 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 0
python main.py --dataset cifar10 --dir-alpha 100 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 42
python main.py --dataset cifar10 --dir-alpha 100 --where-loading init --model cnn -T 10000 --eval-every 100 --C-t 5 --algorithm fedgloss --server-opt sgd --server-lr 1 --server-momentum 0 --batch-size 64 -E 1 --lr 0.01 --weight-decay 0.0004 --momentum 0 --device cuda:0 --rho-l 0.15 --eta 0 --beta 10 --rho 0.15 --T-s 2000 --plots --seed 200299
