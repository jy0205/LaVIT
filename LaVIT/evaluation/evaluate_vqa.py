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
import logging
from copy import deepcopy
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
from models import build_model, LaVITImageProcessor, LaVITQuestionProcessor
from utils import get_rank, save_result, is_main_process
from evaluation.vqa_tools.vqa import VQA
from evaluation.vqa_tools.vqa_eval import VQAEval


try:
    import spacy
    lemmatizer = spacy.load("en_core_web_sm")
except ImportError:
    logging.error(
        """
        Please install spacy and en_core_web_sm model to apply lemmatization.
        python -m spacy download en_core_web_sm
        OR
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        """
    )
    exit(1)


class VQADataset(Dataset):
    def __init__(self, anno_path, vis_root, resolution=224):
        super().__init__()

        with open(anno_path, 'r') as fr:
            self.annotation = json.load(fr)

        self.image_processer = LaVITImageProcessor(image_size=resolution)
        self.text_processer = LaVITQuestionProcessor()
        self.vis_root = vis_root

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann['image']
        image_path = os.path.join(self.vis_root, image_path)
        image = Image.open(image_path).convert("RGB")

        image = self.image_processer(image)
        question = self.text_processer(ann["question"])

        question_id = ann["question_id"]

        if 'answer' in ann.keys():
            if isinstance(ann['answer'], str):
                answer = ann['answer']
            else:
                assert isinstance(ann['answer'], list)
                answer = '@'.join(ann['answer'])

        else:
            assert 'answers' in ann.keys()
            answer = '@'.join([ans['answer'] for ans in ann['answers']]),

        return image, question, question_id, answer

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
        questions = list(transposed_batch[1])
        question_ids = list(transposed_batch[2])
        answers = list(transposed_batch[3])
        return {'image' : images, 'text_input' : questions, 'question_id' : question_ids, 'gt_answer' : answers}

    vis_root = args.imagepath
    dataset = VQADataset(anno_path, vis_root, resolution=args.resolution)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, 
        sampler=sampler, shuffle=False, collate_fn=collate_fn, drop_last=False
    )
    return loader


def lemmatize(answer):
    # Lemmatize the answer str to support equal evaluation
    doc = lemmatizer(answer)

    words = []
    for token in doc:
        if token.pos_ in ["NOUN", "VERB"]:
            words.append(token.lemma_)
        else:
            words.append(token.text)

    answer = " ".join(words)

    return answer


def report_vqa_metrics(result_file, ques_file, anno_file):
    """
    Use official VQA evaluation script to report metrics.
    """
    metrics = {}
    vqa = VQA(anno_file, ques_file)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_file
    )

    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    print("Start VQA evaluation.")
    matched, mismatched = vqa_scorer.evaluate()

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]
    metrics["agg_metrics"] = overall_acc

    print("Overall Accuracy is: %.02f\n" % overall_acc)
    print("Per Answer Type Accuracy is the following:")

    for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        print(
            "%s : %.02f"
            % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
        )
        metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

    return metrics


def report_gqa_metrics(result_file):
    results = json.load(open(result_file, "r"))
    acc = []
    vqa_tool = VQAEval()

    for res in results:
        gt_ans = res["gt"]
        pred = res["answer"]
        pred = vqa_tool.processPunctuation(pred)
        pred = vqa_tool.processDigitArticle(pred)
        pred = vqa_tool.process_ans(pred)
        vqa_acc = 0 if pred != gt_ans else 1
        acc.append(vqa_acc)

    accuracy = sum(acc) / len(acc) * 100
    metrics = {"agg_metrics": accuracy, "acc": accuracy}
    print(metrics)
    return metrics


def eval_vqa(args):
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
        if dataset_name == 'vizwiz':
            # For VizWiz Benchmarks, first judge if the question is answerable
            stage1_prompt = 'Is the question: "{}" answerable?'
            samples_stage1 = deepcopy(samples)
            questions = samples['text_input']
            questions_stage1 = [stage1_prompt.format(text) for text in questions]
            samples_stage1['text_input'] = questions_stage1
            judgements = model.predict_answers(samples_stage1)

        answers = model.predict_answers(samples)

        if dataset_name == 'vizwiz':
            for idx, judge in enumerate(judgements):
                if 'no' in judge.lower():
                    answers[idx] = 'unanswerable'

        question_ids = samples["question_id"]
        questions = samples['text_input']
        gt_answers = samples['gt_answer']

        if dataset_name == 'gqa':
            gt_answers = [lemmatize(gt_answer) for gt_answer in gt_answers]

        for answer, ques_id, ques, gt_answer in zip(answers, question_ids, questions, gt_answers):
            ques_id = int(ques_id)
            results.append({"question_id": ques_id, "answer": answer, "question": ques, "gt": gt_answer,})

    torch.distributed.barrier()

    eval_result_file = save_result(
        result=results,
        result_dir='tmp/vqa',
        filename=f"{dataset_name}_vqa",
        remove_duplicate="question_id",
    )

    if dataset_name == 'gqa':
        if is_main_process():
            report_gqa_metrics(eval_result_file)
    else:
        if dataset_name == 'vqav2':
            anno_file = 'data/coco/annotations/v2_mscoco_val2014_annotations_lemm.json'
            ques_file = 'data/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json'
        elif dataset_name == 'okvqa':
            anno_file = 'data/okvqa/annotations/mscoco_val2014_annotations_lemm.json'
            ques_file = 'data/okvqa/annotations/OpenEnded_mscoco_val2014_questions.json'
        elif dataset_name == 'vizwiz':
            anno_file = 'data/vizwiz/annotations/full_set/vizwiz_val_annotations_lemm.json'
            ques_file = 'data/vizwiz/annotations/full_set/vizwiz_val_questions.json'
        else:
            raise NotImplementedError(f"Not Supported dataset {dataset_name}")
        
        if is_main_process():
            report_vqa_metrics(eval_result_file, ques_file, anno_file)
    
    torch.distributed.barrier()