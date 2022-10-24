#!/bin/bash
python mask_graph_runner.py --cuda 1 --device-index 0 \
                --dgl-format 1 --apply-scheduler 0 --num-node-types 1\
                --func ffd --tree-aggr max-pooling --graph-aggr max-pooling\
                --pos-encoding 0\
                --config-path configs/config_mask_java.yml --num-workers 8\
                --task summarization\
                --parser-path language/java.so --language java\
                --dataset java\
                --apply-copy 0