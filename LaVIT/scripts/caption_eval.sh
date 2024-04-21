#!/bin/bash

# Nocaps
# dataset_name="nocaps"
# imagepath="data/nocaps/images"
# annopath="data/nocaps/annotations/nocaps_val.json"

# Flickr
# dataset_name="flickr"
# imagepath="data/flickr30k/images"
# annopath="data/flickr30k/annotations/flickr30k_test.json"

# COCO
dataset_name="coco"
imagepath="data/coco/images"
annopath="data/coco/annotations/coco_karpathy_test.json"


python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --task_name caption \
    --batch_size 2 \
    --resolution 224 \
    --imagepath $imagepath \
    --annopath $annopath \
    --dataset $dataset_name