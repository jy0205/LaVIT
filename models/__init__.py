from .lavit_for_generation import LaVITforGeneration
from .lavit_for_understanding import LaVITforUnderstanding
from .transform import LaVITImageProcessor, LaVITQuestionProcessor
from utils import convert_weights_to_bf16, convert_weights_to_fp16
from huggingface_hub import snapshot_download


# Building the Model
def build_model(
    model_path='./',
    model_dtype='bf16',
    device_id=None,
    image_size=224,
    use_xformers=False,
    understanding=True,
    load_tokenizer=True,
    pixel_decoding='highres',
    check_safety=True,
    local_files_only=False,
):
    """
    model_path (str): The local directory for the saving the model weight
    model_dtype (str): The precision dtype of the model in inference, bf16 or fp16
    device_id (int): Specifying the GPU ID to loading the model
    use_xformers (bool): default=False, If set True, use xformers to save the GPU memory in the eva clip
    understanding (bool): If set True, use LaVIT for multi-modal understanding, else used for generation
    load_tokenizer (bool): Whether to load the tokenizer encoder during the image generation. For text-to-image generation,
        The visual tokenizer is not needed, set it to `False` for saving the GPU memory. When using for the 
        multi-modal synthesis (the input image needs to be tokenizd to dircrete ids), the load_tokenizer must be set to True.
    pixel_decoding (str): [highres | lowres]: default is `highres`: using the high resolution decoding 
        for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
    check_safety (bool): Should be set to True to enable the image generation safety check
    local_files_only (bool): If you have already downloaded the LaVIT checkpoint to the model_path, 
    set the local_files_only=True to avoid loading from remote
    """
    # Downloading the model checkpoint from the huggingface remote
    print("Downloading the LaVIT checkpoint from huggingface")

    if not local_files_only:
        snapshot_download("rain1011/LaVIT-7B-v1", local_dir=model_path, 
            local_files_only=local_files_only, local_dir_use_symlinks=False)

    if understanding:
        lavit = LaVITforUnderstanding(model_path=model_path, model_dtype=model_dtype, 
                device_id=device_id, use_xformers=use_xformers)
    else:
        lavit = LaVITforGeneration(model_path=model_path, model_dtype=model_dtype, device_id=device_id, 
                use_xformers=use_xformers, check_safety=check_safety, load_tokenizer=load_tokenizer, pixel_decoding=pixel_decoding)

    # Convert the model parameters to the defined precision
    if model_dtype == 'bf16':
        convert_weights_to_bf16(lavit)
    if model_dtype == 'fp16':
        convert_weights_to_fp16(lavit)

    lavit = lavit.eval()

    return lavit