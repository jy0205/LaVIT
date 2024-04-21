from .video_lavit_for_generation import VideoLaVITforGeneration
from .video_lavit_for_understanding import VideoLaVITUnderstandingRunner
from .transform import LaVITImageProcessor
from .video_detokenizer import build_video_detokenizer
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
    local_files_only=False,
    model_sub_dir='language_model',
    max_video_clips=16,
):
    """
    model_path (str): The local directory for the saving the model weight
    model_dtype (str): The precision dtype of the model in inference, bf16 or fp16
    device_id (int): Specifying the GPU ID to loading the model
    use_xformers (bool): default=False, If set True, use xformers to save the GPU memory in the eva clip
    understanding (bool): If set True, use LaVIT for multi-modal understanding, else used for generation
    pixel_decoding (str): [highres | lowres]: default is `highres`: using the high resolution decoding 
        for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
    local_files_only (bool): If you have already downloaded the LaVIT checkpoint to the model_path, 
    set the local_files_only=True to avoid loading from remote
    """

    if understanding:
        video_lavit = VideoLaVITUnderstandingRunner(model_path=model_path, model_dtype=model_dtype, device_id=device_id, 
                            use_xformers=use_xformers, max_clips=max_video_clips)
    else:
        video_lavit = VideoLaVITforGeneration(model_path=model_path, model_dtype=model_dtype, device_id=device_id, 
                                use_xformers=use_xformers, model_sub_dir=model_sub_dir,)
        # Convert the model parameters to the defined precision
        if model_dtype == 'bf16':
            convert_weights_to_bf16(video_lavit)
        if model_dtype == 'fp16':
            convert_weights_to_fp16(video_lavit)
            
        video_lavit = video_lavit.eval()

    return video_lavit