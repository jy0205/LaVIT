import re
import os
import cv2
import math
import torch
import random
import numpy as np
import subprocess

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from mvextractor.videocap import VideoCap
from diffusers.image_processor import VaeImageProcessor


def extract_motions(video_path, raw_file=True, temp_dir=None, fps=6, rescale=False):
    # First dump the video into 6-fps clips
    if raw_file:
        video_name = os.path.split(video_path)[-1].split('.')[0]
        temp_video_path = f'{temp_dir}/{video_name}.mp4'
        cmd = f'/home/jinyang06/ffmpeg/bin/ffmpeg -threads 8 -loglevel error -y -i {video_path} -filter:v fps={fps} -b:v 8000k -c:v mpeg4 -f rawvideo {temp_video_path}'
        ret = subprocess.run(args=cmd, shell=True, timeout=2000)
        if ret.returncode != 0:
            raise RuntimeError(f"Dump video to {fps} ERROR")
    else:
        temp_video_path = video_path

    # The motion vector mean and std
    mean = np.array([[0.0, 0.0]], dtype=np.float64)
    std =  np.array([[0.0993703, 0.1130276]], dtype=np.float64)

    # Rescale for the input of motion tokenizer
    if rescale:
        std = std / 10.0

    # Load motion vector from raw video
    cap = VideoCap()
    ret = cap.open(temp_video_path)
    frames, motions, frame_types = [], [], []

    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        mv = np.ones((h,w,2)) * -10000   # The
        position = motion_vectors[:,5:7].clip((0,0),(w-1,h-1))
        mvs = motion_vectors[:,0:1] * motion_vectors[:,7:9] / motion_vectors[:, 9:]

        # Normalize the motion vector with resoultion
        mvs[:, 0] = mvs[:, 0] / w
        mvs[:, 1] = mvs[:, 1] / h
        # Normalize the motion vector
        mvs = (mvs - mean) / std

        mv[position[:,1],position[:,0]] = mvs
        motions.append(mv)
        frame_types.append(frame_type)
        frames.append(frame[:, :, ::-1])

    return frames, motions, frame_types


class MotionVectorProcessor:

    def __init__(self, width, height):
        transform_list = [
            transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        ]
        self.motion_transform = transforms.Compose(transform_list)

    def __call__(self, motions):
        h, w = motions.shape[-2:]
        pad_h_size = math.ceil(h / 16) * 16
        pad_w_size = math.ceil(w / 16) * 16

        padding_h = pad_h_size - h
        padding_w = pad_w_size - w

        pad = (0,padding_w,0,padding_h)
        motions = torch.nn.functional.pad(motions, pad, mode="constant", value=-10000)
        motions = torch.nn.functional.max_pool2d(motions, kernel_size=16, stride=16)
        motions[motions < -1000] = 0.0

        # interpolate the 13-th frame, which is I frame, don't have a motion
        motions_to_inter = torch.cat([motions[11:12], motions[13:14]])
        motions_to_inter = motions_to_inter.permute(1, 0, 2, 3).unsqueeze(0)
        motions_to_inter = torch.nn.functional.interpolate(motions_to_inter, scale_factor=(1.5, 1.0, 1.0), mode='trilinear')[0]
        motions_to_inter = motions_to_inter.permute(1, 0, 2, 3)
        motions[12] = motions_to_inter[1]

        motion_vectors = self.motion_transform(motions)   # [T, C, H, W]
        motion_vectors = motion_vectors.permute(1, 0, 2, 3)  # From [T, C, H, W] => [C, T, H, W]

        return motion_vectors


class LaVITEvalVideoProcessor:
    # For custom video understanding
    def __init__(self, image_size=224, num_frames=24, fps=6, max_clips=8, temp_dir=None):
        # fps=6
        self.max_frames = num_frames
        self.normalize_resolution = True
        self.fps = fps
        self.max_clips = max_clips

        self.motion_transform = MotionVectorProcessor(width=36, height=20)
        self.image_transform = LaVITImageProcessor(image_size=image_size, is_train=False)

        self.temp_dir = './tmp'
        # Used for temporally save the reencoded video
        if temp_dir is not None:
            self.temp_dir = temp_dir

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def sample_video_clips(self, video_path, raw_file=True, temp_dir=None):
        # First dump the video into 6-fps clips
        if temp_dir is None:
            temp_dir = self.temp_dir

        frames, motions, frame_types = extract_motions(video_path, raw_file, temp_dir, fps=self.fps, rescale=True)

        # Next, sample the video clips from a long video
        total_frames = len(frame_types)
        start_indexs = np.where(np.array(frame_types)=='I')[0]
        
        if len(start_indexs) == 0:
            raise ValueError(f"Empty Start indexs: {video_path}")

        # Only select the I-frame that follows 11 P-frame
        if len(start_indexs) > 1:
            end_indexs = start_indexs + 12
            filter_start_indexs = start_indexs[:-1][end_indexs[:-1] == start_indexs[1:]]    
        else:
            filter_start_indexs = start_indexs

        # FIlter the frames that exceed the max frames
        filter_start_indexs = filter_start_indexs[filter_start_indexs + self.max_frames <= total_frames]

        # Only if the remained frame number >= 2, we use it
        if len(filter_start_indexs) >= 2:
            start_indexs = filter_start_indexs

        if len(start_indexs) / 2 <= self.max_clips:
            if len(start_indexs) > 2:
                selected_index = start_indexs[1::2]
            else:
                selected_index = start_indexs[0::2]
        else:
            start_indexs = start_indexs[1::2]   # We sample every two I-frames, skip the first I-frame
            if len(start_indexs) > self.max_clips:
                selected_index = np.linspace(0, len(start_indexs)-1, self.max_clips, dtype=int)
                selected_index = start_indexs[selected_index]
            else:
                selected_index = start_indexs

        assert len(selected_index) <= self.max_clips
        key_frame_indices = list(selected_index)
        video_motion_sequences = []
        video_frame_sequences = []

        for fid in key_frame_indices:
            video_frame_sequences.append(Image.fromarray(frames[fid]).convert("RGB"))
            frame_indices = np.arange(fid, min(fid + self.max_frames, total_frames))
            clip_motions = [torch.from_numpy(motions[i].transpose((2,0,1))) for i in frame_indices]
            clip_motions = torch.stack(clip_motions).float()
            if clip_motions.shape[0] < self.max_frames:
                pad_clip_motions = torch.ones((self.max_frames, 2, clip_motions.shape[-2], clip_motions.shape[-1])) *  -10000
                pad_clip_motions[:len(clip_motions)] = clip_motions
                clip_motions = pad_clip_motions
            video_motion_sequences.append(clip_motions)

        return video_frame_sequences, video_motion_sequences

    def __call__(self, video_path, raw_file=True, use_cache=False, temp_dir=None):
        try:
            video_frame_sequences, video_motion_sequences = self.sample_video_clips(video_path, raw_file, temp_dir)

            video_motion_vectors = []
            for clip_motions in video_motion_sequences:
                clip_motion_vectors = self.motion_transform(clip_motions)
                video_motion_vectors.append(clip_motion_vectors)

            video_motion_vectors = torch.stack(video_motion_vectors)

            video_visual_vectors = [self.image_transform(frame) for frame in video_frame_sequences]
            video_visual_vectors = torch.stack(video_visual_vectors)

            assert video_visual_vectors.shape[0] == video_motion_vectors.shape[0]
            assert video_motion_vectors.shape[1] == 2
            assert video_motion_vectors.shape[2] == 24

            return video_visual_vectors, video_motion_vectors

        except Exception as e:
            raise ValueError(f"load video motion error {e}")


class LaVITImageProcessor:
    def __init__(self, image_size=224, is_train=False):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        if is_train:
            transform_list = [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.CenterCrop(image_size),
                transforms.Normalize(mean, std)  # assume image
            ]

        else:
            transform_list = [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]

        self.transform = transforms.Compose(transform_list)

    def __call__(self, item):
        return self.transform(item)