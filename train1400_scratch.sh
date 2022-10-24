#!/bin/bash
python graph_runner.py --cuda 1 --device-index 3\
                --dgl-format 1 --apply-scheduler 0 --num-node-types 1\
                --func gru --tree-aggr max-pooling --graph-aggr avg-pooling\
                --pos-encoding 0\
                --config-path configs/config1400_scratch.yml --num-workers 4\ 
                --task classification\
                --dataset cpp --parser-path language/cpp.so\
                --file-ext cpp --language cpp --test 0

