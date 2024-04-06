#!/bin/bash
export PYSERINI_CACHE=/store2/scratch/s8sharif

# TODO: make the bash file work with flag values for different runs
bs=512
retrieved_results_path=dataset/blip/retrieved.jsonl
caption_count=$(wc -l < $retrieved_results_path)
echo $caption_count
k=5




for ((i = 0; i < $caption_count; i += $bs)); do
    next_index=$((i + bs))
    CUDA_VISIBLE_DEVICES=0 python txt_to_img_inference.py --prompt_file txt-to-img-prompt-with-examples.txt \
    --k=$k --model_name lavit \
    --output_dir /store2/scratch/s8sharif/UniIR/txt_to_img_llm_outputs \
    --retrieved_results_path $retrieved_results_path \
    --index $i"_"$next_index --retriever_name "BLIP_FF"
done
