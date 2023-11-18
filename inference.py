import os
import torch
import random
import torch.nn as nn
import numpy as np
from models import build_model
from PIL import Image


def example_for_understanding():
    # Building model and load weight
    model = build_model(model_path=model_path, model_dtype=model_dtype,
                device_id=device_id, use_xformers=True, understanding=True)
    model = model.to(device)    

    # Image Captioning
    image_path = 'demo/caption_image.jpg'
    caption = model.generate({"image": image_path})[0]
    print(caption)

    # Visual Question Answering
    image_path = 'demo/qa_image.jpg'
    question = "What's that drink in the glass?"
    print("Question:", question)
    answer = model.predict_answers({"image": image_path, "text_input": question}, max_len=10)[0]
    print("The answer is: ", answer)


def example_for_generation():
    # Building model and load weight
    model = build_model(model_path=model_path, model_dtype=model_dtype, check_safety=False,
                device_id=device_id, use_xformers=True, understanding=False)
    model = model.to(device)    

    # LaVIT support 6 different image aspect ratios
    ratio_dict = {
        '1:1' : (1024, 1024),
        '4:3' : (896, 1152),
        '3:2' : (832, 1216),
        '16:9' : (768, 1344),
        '2:3' : (1216, 832),
        '3:4' : (1152, 896),
    }

    # The image aspect ratio you want to generate
    ratio = '1:1'
    height, width = ratio_dict[ratio]

    # Text-to-Image Generation
    prompt = "A photo of an astronaut riding a horse in the forest."
    with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        image = model.generate_image(prompt, width=width, height=height, 
            guidance_scale_for_llm=4.0, num_return_images=1)[0]
    image.save("output/t2i_output.jpg")

    # Multi-modal Image synthesis
    image_prompt = 'demo/dog.jpg'
    text_prompt = 'It is running in the snow'
    input_prompts = [(image_prompt, 'image'), (text_prompt, 'text')]
    with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        image = model.multimodal_synthesis(input_prompts, width=width, height=height, 
            guidance_scale_for_llm=5.0, num_return_images=1)[0]
    image.save("output/it2i_output.jpg")


if __name__ == "__main__":
    model_path='models/LaVIT_checkpoint'
    model_dtype='bf16'

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device('cuda')
    torch_dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16

    ## For Multi-Modal Understanding
    example_for_understanding()

    ## For Multi-Modal Generation
    example_for_generation()

