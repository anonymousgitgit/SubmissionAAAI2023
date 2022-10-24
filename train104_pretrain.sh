#!/bin/bash
python graph_runner.py --cuda 1 --device-index 2 --dgl-format 1 --apply-scheduler 1 --num-node-types 1 --parser-path language/c.so --file-ext c --language c\
                --task classification --dataset c \
                --func gru --tree-aggr max-pooling --graph-aggr max-pooling --pos-encoding 0 --scratch-train 0\
                --pretrain-cfg-path configs/config_mask_cpp.yml\
                --config-path configs/config104_pretrain.yml --num-workers 4
