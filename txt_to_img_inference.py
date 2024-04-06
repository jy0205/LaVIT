import argparse
from datetime import datetime
import json
import os
import pathlib
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

import jsonlines
from models import build_model
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import txt_to_img_prompt_generator

# TODO: make these customizable by adding to args
MAX_TOKENS = 300
MBIER_BASE_PATH = "/mnt/users/s8sharif/M-BEIR/"


def build_lavit_model():
    # Todo: make all values configurable
    model_path = "models/lavit"

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # device_id = 2
    # torch.cuda.set_device(device_id)
    device = torch.device("cuda")

    # For Multi-modal Image Generation, must set `load_tokenizer=True` to load the tokenizer to tokenize input image.
    # If you have already install xformers, set `use_xformers=True` to save the GPU memory (Xformers is not supported on V100 GPU)
    # If you have already download the checkpoint, set `local_files_only=True`` to avoid auto-downloading from remote
    model_dtype = "bf16"
    model = build_model(
        model_path=model_path,
        model_dtype=model_dtype,
        check_safety=False,
        use_xformers=False,
        understanding=False,
        load_tokenizer=True,
    )
    model = model.to(device)
    print("Building Model Finsished")
    return model


def infer_lavit(
    p_class: txt_to_img_prompt_generator.Prompt,
    retrieval_dict: Dict[
        str, Tuple[str, List[str]]
    ],  # Dict from captions to retrieved example tuples( qid, list of sample image paths)
    model,
    output_image_dir,
):
    outputs = []
    keys = list(retrieval_dict.keys())
    for i in tqdm(range(0, len(keys)), desc="Generating images"):
        try:
            caption = keys[i]
            qid, sample_images = retrieval_dict[caption]
            txt_prompt = p_class.prepare_message(
                caption=caption, num_sample_images=len(sample_images)
            )
            prompts = []
            for img_path in sample_images:
                prompts.append((os.path.join(MBIER_BASE_PATH, img_path), "image"))
            prompts.append((txt_prompt, "text"))
            # Todo: make params configurable
            height, width = 224, 224
            torch_dtype = torch.bfloat16
            with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
                images = model.multimodal_synthesis(
                    prompts,
                    width=width,
                    height=height,
                    guidance_scale_for_llm=4.0,
                    num_return_images=1,
                    num_inference_steps=25,
                    top_k=50,
                )
        except Exception as e:
            print(f"Exeption: processing {qid} with {prompts} caused {e}")
            continue
        output_img_path = os.path.join(
            output_image_dir, f"{qid}_{datetime.isoformat(datetime.now())}.jpg"
        )
        images[0].save(output_img_path)
        print(f"Processed caption: {caption}")
        print(f"saved generated image at {output_img_path}")
        outputs.append(
            {
                "qid": qid,
                "caption": caption,
                "prompt": prompts,
                "response": output_img_path,
            }
        )
        print("-" * 79)

    return outputs


def load_retrieval_dict(file_path, index, k):
    # Storing only relevant retrieval info
    retrieval_dict = {}
    if index == "full":
        start = 0
        end = None
    else:
        temp = index.split("_")
        start = int(temp[0])
        end = int(temp[1])
    index = 0
    with jsonlines.open(file_path) as reader:
        for obj in tqdm(reader, desc="Reading queries"):
            if end and index == end:
                break
            if index < start:
                index += 1
                continue
            caption = obj["query"]["query_txt"]
            if caption:
                candidates = []
                for cand in obj.get("candidates")[0:k]:
                    if cand["img_path"]:
                        candidates.append(cand["img_path"])
                retrieval_dict[caption] = (obj["query"]["qid"], candidates)
            index += 1
    return retrieval_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", default=False, help="Prompt file")
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Number of retrieved examples included in the prompt",
    )
    parser.add_argument("--model_name", default="lavit")
    parser.add_argument(
        "--index", default="full", help="Add start end indices in x_y format"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory to store llm outputs, the output dir would be: output_dir/'model_name'_outputs/'retriever_name'_k",
    )
    parser.add_argument(
        "--retrieved_results_path",
        required=True,
        help="path to the jsonl file containing query + candidates pairs",
    )
    parser.add_argument(
        "--retriever_name",
        required=True,
        help="Name of the retriever that has retrieved input candidates",
    )
    args = parser.parse_args()
    if args.k == 0 and "-with-" in args.prompt_file:
        raise ValueError("Invalid template file for zero-shot inference.")
    elif args.k > 0 and "-without-" in args.prompt_file:
        raise ValueError("Invalid template file for few-shot inference.")

    infer_mapping = {"lavit": infer_lavit}

    # Storing only relevant retrieval info
    retrieval_dict = load_retrieval_dict(
        args.retrieved_results_path, args.index, args.k
    )

    model = args.model_name
    if model == "lavit":
        model = build_lavit_model()
    p_class = txt_to_img_prompt_generator.Prompt(args.prompt_file, args.k)
    result_dir = os.path.join(
        args.output_dir,
        f"{args.model_name}_outputs",
        f"{args.retriever_name}_k{args.k}",
    )
    output_image_dir = os.path.join(result_dir, "generated_images")
    os.makedirs(output_image_dir, exist_ok=True)
    result = infer_mapping[args.model_name](
        p_class, retrieval_dict, model, output_image_dir
    )
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(
        result_dir, f"{args.index}_{datetime.isoformat(datetime.now())}.json"
    )
    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


if __name__ == "__main__":
    main()
