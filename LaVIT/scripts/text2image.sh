#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 evaluate.py \
    --task_name text2image \
    --annopath data/coco/annotations/coco_t2i_eval.json \
    --imagepath data/coco/images \
    --batch_size 2 \
    --resolution 512