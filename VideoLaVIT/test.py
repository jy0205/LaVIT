from models import build_video_detokenizer
from IPython import embed
import torch
from collections import OrderedDict

# model_path = '/home/jinyang06/models/VideoLaVIT_release'
# model = build_video_detokenizer(model_path)
# embed()


# model_path = '/etc/ssd2/jinyang06/video_lavit/detokenizer/sdxl_720p_cb1024_no_ft/checkpoint-8000/pytorch_model.bin'
# model_path = '/etc/ssd2/jinyang06/video_lavit/detokenizer/scratch_720p_raw/checkpoint-13000/pytorch_model.bin'
model_path = '/etc/ssd2/jinyang06/video_lavit/detokenizer/scratch_320p_raw/checkpoint-26000/pytorch_model.bin'
model = torch.load(model_path, map_location='cpu')
new_model = OrderedDict()

for key in model.keys():
    if key.startswith('motion_tokenizer.') or key.startswith('vae'):
        continue
    else:
        # if key.startswith('unet.'):
        #     new_key = '.'.join(key.split('.')[1:])
        #     new_model[new_key] = model[key]
        # else:
        #     new_model[key] = model[key]
        new_model[key] = model[key]

torch.save(new_model, '/home/jinyang06/models/VideoLaVIT_release/video_3d_unet.bin')
# torch.save(new_model, "/home/jinyang06/models/VideoLaVIT_release/video_detokenizer/unet/diffusion_pytorch_model.bin")