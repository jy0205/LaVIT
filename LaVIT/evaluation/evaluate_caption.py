import os
import sys
import argparse

import json
import math
import random
import datetime
import torch
import torch.distributed as dist

from torch import nn
from torchvision import transforms as pth_transforms
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch.backends.cudnn as cudnn

from PIL import Image
from functools import partial
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from models import build_model, LaVITImageProcessor
from utils import get_rank, save_result, is_main_process

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from IPython import embed


class CaptionDataset(Dataset):
    def __init__(self, anno_path, vis_root, resolution=224):
        super().__init__()

        with open(anno_path, 'r') as fr:
            self.annotation = json.load(fr)

        self.transform = LaVITImageProcessor(image_size=resolution)
        self.vis_root = vis_root

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['image']
        image_path = os.path.join(self.vis_root, image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if "img_id" in ann:
            img_id = ann["img_id"]
        else:
            img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return image, img_id

    def __len__(self):
        return len(self.annotation)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    args.dist_url = "env://"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def build_data_loader(args):
    anno_path = args.annopath

    def collate_fn(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], dim=0)
        image_ids = list(transposed_batch[1])
        return {'image' : images, 'image_id' : image_ids}

    vis_root = args.imagepath
    dataset = CaptionDataset(anno_path, vis_root, resolution=args.resolution)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, 
        sampler=sampler, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return loader


def report_caption_metrics(annotation_file, results_file):
    print(f"The annotation file is {annotation_file}")
    
    coco = COCO(annotation_file)

    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")


def eval_caption(args):
    # Evaluation
    init_distributed_mode(args)

    seed = 42 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    model_path = args.model_path
    model_dtype = args.model_dtype
    torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
    dataset_name = args.dataset

    model = build_model(model_path=model_path, model_dtype=model_dtype,
                local_files_only=True, use_xformers=False, understanding=True)
    device = torch.device('cuda')
    model = model.to(device)

    data_loader = build_data_loader(args)
    rank = args.rank
    results = []

    for samples in tqdm(data_loader):
        image_ids = samples['image_id']
        captions = model.generate(samples)
        for caption, img_id in zip(captions, image_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

    torch.distributed.barrier()

    eval_result_file = save_result(
        result=results,
        result_dir='tmp/caption',
        filename=f"{dataset_name}_caption",
        remove_duplicate="image_id",
    )

    if dataset_name == 'coco':
        annotation_file = 'data/coco_gt/coco_karpathy_test_gt.json'
    elif dataset_name == 'nocaps':
        annotation_file = 'data/nocaps/nocaps_val_4500_captions.json'
    elif dataset_name == 'flickr':
        annotation_file = 'data/flickr30k/flickr30k_test_gt.json'
    else:
        raise NotImplementedError(f"Not Supported dataset {dataset_name}")
    
    if is_main_process():
        report_caption_metrics(annotation_file, eval_result_file)
    
    torch.distributed.barrier()