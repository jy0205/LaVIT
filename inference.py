import os
import torch
import random
import torch.nn as nn
from models import build_model
from PIL import Image


def example_for_understanding():
    # Building model and load weight
    model = build_model(model_path=model_path, model_dtype=model_dtype,
                device_id=device_id, use_xformers=False, understanding=True)
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
    model = build_model(model_path=model_path, model_dtype=model_dtype,
                device_id=device_id, use_xformers=False, understanding=False)
    model = model.to(device)    

    # Text-to-Image Generation
    prompt = "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert."
    with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        image = model.generate_image(prompt, guidance_scale_for_llm=3.0, num_return_images=2)[0]
    image.save("output/i2t_output.jpg")

    # Multi-modal Image synthesis
    image_prompt = 'demo/dog.jpg'
    text_prompt = 'It is running in the snow'
    input_prompts = [(image_prompt, 'image'), (text_prompt, 'text')]
    with torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
        image = model.multimodal_synthesis(input_prompts, guidance_scale_for_llm=5.0, num_return_images=2)[0]
    image.save("output/it2i_output.jpg")


if __name__ == "__main__":
    model_path='/home/jinyang06/models/LaVIT_checkpoint'
    model_dtype='bf16'

    device_id = 0
    torch.cuda.set_device(device_id)
    device = torch.device('cuda')
    torch_dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16

    random.seed(42)
    torch.manual_seed(42)

    ## For Multi-Modal Understanding
    example_for_understanding()

    ## For Multi-Modal Generation
    example_for_generation()

