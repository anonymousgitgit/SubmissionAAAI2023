#!/bin/bash
python clone_astnn_runner.py --cuda 1 --device-index 2\
                --dgl-format 1 --apply-scheduler 1 --num-node-types 1\
                --func gru --tree-aggr max-pooling --graph-aggr max-pooling\
                --pos-encoding 0 --scratch-train 0\
                --pretrain-cfg-path configs/config_mask_cpp.yml\
                --config-path configs/config_clone_astnn_pretrain.yml --num-workers 4\ 
                --task clone\
                --dataset c --parser-path language/c.so\
                --file-ext c --language c
