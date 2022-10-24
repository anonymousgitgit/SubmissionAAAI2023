#!/bin/bash
python graph_runner.py --dataset c --parser-path language/c.so --file-ext c --language c\
                --dgl-format 1 --apply-scheduler 0 --num-node-types 1\
                --func gru --tree-aggr max-pooling --graph-aggr max-pooling\
                --pos-encoding 0\
                --config-path configs/config104_scratch.yml --num-workers 4\ 
                --task classification
