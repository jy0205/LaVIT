#!/bin/bash

# Nocaps
dataset_name="vizwiz"
imagepath="data/vizwiz/images/val"
annopath="data/vizwiz/annotations/full_set/val_answerable.json"

# GQA
dataset_name="gqa"
imagepath="data/gqa/images"
annopath="data/gqa/annotations/testdev_balanced_questions.json"

# OKVQA
dataset_name="okvqa"
imagepath="data/coco/images"
annopath="data/okvqa/annotations/vqa_val_eval.json"

# VQAv2
dataset_name="vqav2"
imagepath="data/coco/images"
annopath="data/coco/annotations/vqa_val_eval.json"


python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --task_name vqa \
    --batch_size 2 \
    --resolution 224 \
    --imagepath $imagepath \
    --annopath $annopath \
    --dataset $dataset_name