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
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

from models.openai_clip import clip
from models import build_model
from utils import get_rank, is_main_process
from IPython import embed


class ImageDataset(Dataset):
    def __init__(self, anno_path, root, resolution=256):
        super().__init__()

        with open(anno_path, 'r') as fr:
            self.annotation = json.load(fr)

        transforms = [
            pth_transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
        ]

        self.transform = pth_transforms.Compose(transforms)
        self.root = root

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['image']
        image_path = os.path.join(self.root, image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        text = ann['text']
        return image, text, index

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


def build_clip_model(args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    device = torch.device('cuda')
    model_path = args.model_path
    clip_model_path = os.path.join(model_path, "openai_clip_ViT-L-14.bin")
    clip_model, _ = clip.load(clip_model_path, device='cpu', jit=False)
    clip_model = clip_model.to(device).eval()
    return clip_model, transform


def build_data_loader(args):
    anno_path = args.annopath

    def collate_fn(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], dim=0)
        texts = list(transposed_batch[1])
        keys = list(transposed_batch[2])
        return {'image' : images, 'key' : keys, 'text' : texts}

    vis_root = args.imagepath
    dataset = ImageDataset(anno_path, vis_root, resolution=args.resolution)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, 
        sampler=sampler, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return loader


def select_base_clip(model, preprocess, raw_images, texts):
    # input images : [bs * num_images]  [bs]
    batch_size = len(texts)
    images = [preprocess(image) for image in raw_images]
    images_num_per_text = len(images) // batch_size
    images = torch.stack(images)
    device = torch.device('cuda')
    # images = torch.reshape(batch_size, -1, images.shape[1:])
    all_texts = []
    for text in texts:
        all_texts.extend([text] * images_num_per_text)
    
    text_tokens = clip.tokenize(all_texts)
    assert text_tokens.shape[0] == images.shape[0]

    with torch.no_grad():
        image_features = model.encode_image(images.to(device))
        text_features = model.encode_text(text_tokens.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        match_scores = (image_features * text_features).sum(dim=-1) # [bs]
    
    match_scores = match_scores.reshape(batch_size, images_num_per_text)  # [bs, 4]
    max_index = match_scores.max(dim=-1)[-1]
    
    output_images = []
    for i, index in enumerate(max_index.tolist()):
        real_index = index + i * images_num_per_text
        output_images.append(raw_images[real_index])
    
    return output_images


def eval_t2i(args):
    # Evaluation
    init_distributed_mode(args)

    seed = 0 + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    model_path = args.model_path
    model_dtype = args.model_dtype
    torch_dtype = torch.bfloat16 if model_dtype=="bf16" else torch.float16
    model = build_model(model_path=model_path, model_dtype=model_dtype, local_files_only=True,
                use_xformers=False, understanding=False, check_safety=False, load_tokenizer=False, pixel_decoding='lowres')
    device = torch.device('cuda')
    model = model.to(device)

    data_loader = build_data_loader(args)
    resolution = args.resolution
    rank = args.rank
    device = torch.device('cuda')

    batch_size = args.batch_size
    fid = FrechetInceptionDistance(normalize=True, sync_on_compute=True, compute_on_cpu=False)
    fid.to(device)

    clip_model, preprocess = build_clip_model(args)
    num_images = 4    # The generated images per prompt

    for sample in tqdm(data_loader):
        real_images = sample['image']
        keys = sample['key']
        prompts = sample['text']

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            raw_images = model.generate_image(prompts, num_inference_steps=50, top_p=1.0, top_k=300, 
                guidance_scale_for_decoder=1.5, length_penalty=1, temperature=1, 
                num_return_images=num_images, guidance_scale_for_llm=1.5,
            )

        # Select the best sample based on the clip score
        output_images = select_base_clip(clip_model, preprocess, raw_images, prompts)
        fake_images = [to_tensor(img) for img in output_images]
        fake_images = torch.stack(fake_images)

        fid.update(real_images.to(device), real=True)
        fid.update(fake_images.to(device), real=False)
        torch.distributed.barrier()

    fid_score = float(fid.compute())
    torch.distributed.barrier()
    print(f"Generate finished, the FID score is {fid_score}")