import os
import argparse
from evaluation import eval_t2i, eval_vqa, eval_caption


def parse_args():
    parser = argparse.ArgumentParser('Evaluation code for LaVIT')
    parser.add_argument('--task_name', default='vqa', type=str, help="The Evaluation Task Name")
    parser.add_argument('--model_path', default='./', type=str, help="model ckpt path")
    parser.add_argument('--model_dtype', default="bf16", type=str, help="The model precision: bf16 or fp16")
    parser.add_argument('--annopath', default='./', type=str, help="The annotation path")
    parser.add_argument('--imagepath', default='./', type=str, help="The image save dir")
    parser.add_argument('--dataset', default='coco', type=str, help="The dataset name")
    parser.add_argument('--batch_size', default=2, type=int, help="The batchsize")
    parser.add_argument('--resolution', default=224, type=int, help="The evaluation resolution")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task_name == 'vqa':
        eval_vqa(args)
    elif args.task_name == 'caption':
        eval_caption(args)
    elif args.task_name == 'text2image':
        eval_t2i(args)