#!/bin/bash
python graph_runner.py --cuda 1 --device-index 1\
                --dgl-format 1 --apply-scheduler 1 --num-node-types 1\
                --func gru --tree-aggr max-pooling --graph-aggr max-pooling\
                --pos-encoding 0 --scratch-train 0\
                --pretrain-cfg-path configs/config_mask_cpp.yml\
                --config-path configs/config1000_pretrain.yml --num-workers 4\ 
                --task classification\
                --dataset cpp --parser-path language/cpp.so\
                --file-ext cpp --language cpp
