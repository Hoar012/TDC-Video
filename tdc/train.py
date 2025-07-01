# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.


import copy
import datetime
import json
import os
import pathlib
import uuid
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional, Sequence

import numpy as np

import torch

import transformers

from decord import cpu, VideoReader
# from petrel_client.client import Client
from io import BytesIO

import sys
sys.path.append(os.getcwd())
from tdc import conversation as conversation_lib

from tdc.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from tdc.language_model.cambrian_llama import CambrianLlamaForCausalLM
from tdc.language_model.cambrian_qwen import CambrianQwenForCausalLM
from tdc.mm_datautils import (
    preprocess,
    preprocess_multimodal,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
    find_all_linear_names
)
from tdc.mm_trainer import LLaVATrainer
from PIL import Image, ImageSequence

from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch import distributed as dist

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

from transformers.integrations import TensorBoardCallback

TENSORBOARD_LOG_DIR_NAME: str = "tensorboard_logs"
from tdc.audio_models.processor import Processor

@dataclass
class ModelArguments:
    input_model_filename: Optional[str] = field(default=None)
    output_model_filename: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    grid_size: Optional[int] = field(default=8)
    vision_tower_type: Optional[str] = field(default="sam")
    mm_hidden_size: Optional[int] = field(default=256)

    # cambrian
    vision_tower_aux_list: Optional[str] = field(
        default='["siglip/CLIP-ViT-SO400M-14-384", "facebook/dinov2-giant-res378"]'
    )
    vision_tower_aux_token_len_list: Optional[str] = field(default="[576, 576]")
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default="[576]")
    connector_depth: Optional[int] = field(default=3)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=0)
    stride_of_vision_sampler_layers: Optional[int] = field(default=3)

    is_st_sampler: bool = field(default=False)
    highres_connect: bool = field(default=False)
    highres: bool = field(default=False)
    connect_layer: Optional[int] = field(default=2)
    lowres_token: Optional[int] = field(default=8)
    dino_threshold: float = field(default=0.9)
    drop_threshold: float = field(default=0.8)
    frame_pos: bool = field(default=False)
    is_image_newline: bool = field(default=True)

    ## New
    pretrained_qformer: Optional[str] = field(default=None)
    unfreeze_mm_compressor: Optional[bool] = field(default=True)
    audio_input: Optional[bool] = field(default=False)
    unfreeze_audio_encoder: Optional[bool] = field(default=False)
    max_num_segments: Optional[int] = field(default=24)
    text_input: Optional[bool] = field(default=True)
    query_type: Optional[str] = field(default="Avg_pool")
    context_token_num: Optional[int] = field(default=16)
    add_static: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_position: Optional[int] = field(default=91)
    image_folder: Optional[str] = field(default=None)
    uniform_sample: bool = field(default=False)
    image_aspect_ratio: str = "square"
    num_points: int = field(default=0)
    video_fps: float = field(default=1)
    use_subtitle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_text_decoder: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    unfreeze_mm_image_decoder: bool = field(default=False)

    mm_vision_sampler_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    model_max_length: Optional[int] = field(default=8192)

    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side + 1, dtype=torch.bool
        )
    else:
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side, dtype=torch.bool
        )
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset(
        (num_tokens_per_side, num_tokens_per_side), image_size
    )
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset - 1 : -1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0) - 1
    return attention_mask, position_ids


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def prepare_multimodal_data(
    input_ids,  # pyre-fixme[2]
    labels,  # pyre-fixme[2]
    attention_mask,  # pyre-fixme[2]
    image_sizes,  # pyre-fixme[2]
    image_token_len=576,  # pyre-fixme[2]
    image_aux_token_len_list=[192 * 192],  # pyre-fixme[2]
    max_length=2048,  # pyre-fixme[2]
):
    input_ids_im_replaced = []
    labels_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
    base_image_token_len_per_side = int(image_token_len**0.5)
    image_aux_token_len_per_side_list = [
        int(image_aux_token_len_per_side**0.5)
        for image_aux_token_len_per_side in image_aux_token_len_list
    ]
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        assert num_images == 1, num_images
        image_size = image_sizes[batch_idx]

        image_token_indices = (
            [-1]
            + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []

        cur_labels = labels[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(
                cur_input_ids[
                    image_token_indices[i] + 1 : image_token_indices[i + 1] + 1
                ]
            )
            cur_labels_im_replaced.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_attention_mask_im_replaced.append(
                cur_attention_mask[
                    image_token_indices[i] + 1 : image_token_indices[i + 1]
                ]
            )
            cur_position_ids_im_replaced.append(
                torch.arange(
                    index,
                    index + image_token_indices[i + 1] - (image_token_indices[i] + 1),
                    dtype=torch.long,
                    device=cur_input_ids.device,
                )
            )
            index += image_token_indices[i + 1] - (image_token_indices[i] + 1)

            if i < len(image_token_indices) - 2:
                num_tokens_per_side = int(image_token_len**0.5)
                image_token_len_with_newline = image_token_len + num_tokens_per_side
                cur_input_ids_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline - 1,),
                        0,
                        device=cur_input_ids.device,
                        dtype=cur_input_ids.dtype,
                    )
                )
                cur_labels_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline,),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

                cur_im_attention_mask, cur_im_position_ids = prepare_image_info(
                    image_size, image_token_len, newline=True
                )

                for aux_i, image_aux_token_len_per_side in enumerate(
                    image_aux_token_len_per_side_list
                ):
                    assert image_aux_token_len_per_side >= base_image_token_len_per_side
                    num_base_crops_per_aux_side = (
                        image_aux_token_len_per_side // base_image_token_len_per_side
                    )

                    cur_im_aux_attention_mask, _ = prepare_image_info(
                        image_size, image_aux_token_len_per_side**2
                    )
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.view(
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                    )
                    cur_im_aux_attention_mask = (
                        cur_im_aux_attention_mask.permute(0, 2, 1, 3)
                        .contiguous()
                        .flatten(0, 1)
                        .flatten(1, 2)
                    )
                    cur_im_aux_attention_mask[
                        cur_im_aux_attention_mask.sum(dim=1) == 0
                    ] = True
                    im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
                cur_im_position_ids += index

                if cur_attention_mask[image_token_indices[i + 1]]:
                    cur_attention_mask_im_replaced.append(cur_im_attention_mask)
                    cur_position_ids_im_replaced.append(
                        cur_im_position_ids.to(torch.long)
                    )
                    index = cur_im_position_ids.max() + 1
                else:
                    num_tokens_per_side = int(image_token_len**0.5)
                    image_token_len_with_newline = image_token_len + num_tokens_per_side
                    cur_attention_mask_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_attention_mask.device,
                            dtype=cur_attention_mask.dtype,
                        )
                    )
                    cur_position_ids_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_input_ids.device,
                            dtype=torch.long,
                        )
                    )

        input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
        labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
        attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
        position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))

    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = torch.stack(new_input_ids)
    new_labels = torch.stack(new_labels)
    new_attention_mask = torch.stack(new_attention_mask)
    new_position_ids = torch.stack(new_position_ids)
    im_aux_attention_masks_list = [
        torch.stack(im_aux_attention_masks)
        for im_aux_attention_masks in im_aux_attention_masks_list
    ]
    return (
        new_input_ids,
        new_labels,
        new_attention_mask,
        new_position_ids,
        im_aux_attention_masks_list,
    )

def uniform_sample(image, num_sample=200):
    sample_indices = torch.ones(len(image), dtype= torch.int16)

    if len(image) > num_sample:
        interval = len(image) / float(num_sample)
        indices = [int(interval * i) for i in range(num_sample)]
        image = [image[idx] for idx in indices]
        sample_indices[indices] -= 1
        sample_indices = 1 - sample_indices
    return image, sample_indices

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        # pyre-fixme[2]: Parameter must be annotated.
        data_args,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        # pyre-fixme[4]: Attribute must be annotated.
        self.list_data_dict = list_data_dict
        self.data_path = data_path
        # pyre-fixme[4]: Attribute must be annotated.
        self.data_args = data_args
        # pyre-fixme[4]: Attribute must be annotated.
        self.length = self._get_length()
        
        # self.client = Client()

    # pyre-fixme[3]: Return type must be annotated.
    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, "r") as file:
            for i, _ in enumerate(file):
                pass
        return i + 1  # pyre-fixme

    def __len__(self) -> int:
        return len(self.list_data_dict)

    # pyre-fixme[3]: Return type must be annotated.
    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, "length_list") and hasattr(self, "modality_length_list"):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list  # pyre-fixme

        self.length_list = []
        self.modality_length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                self.data_args.image_token_len if self._has_image(sample) else 0
            )
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            self.length_list.append(cur_len + img_tokens)
            modality_len = cur_len if "image" in sample else -cur_len
            self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:  # pyre-fixme
        if "image" in sample and not str(sample["image"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        if "video" in sample and not str(sample["video"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        return False
    
    def _has_audio(self, sample: dict) -> bool:  # pyre-fixme
        if "audio" in sample and not str(sample["audio"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        return False

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        dat = sources
        # print("index:", i, dat)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)
        has_audio = self._has_audio(dat)
        if has_image:
            if "image" in dat:
                image_file = dat["image"]
                image_folder = self.data_args.image_folder
                processor_aux_list = self.data_args.image_processor_aux_list
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
                except:
                    print(
                        "Not exist: ",
                        os.path.join(image_folder, image_file),
                        flush=True,
                    )
                    return self.__getitem__(0)
                image_size = image.size
            else:
                video_file = dat["video"]
                processor_aux_list = self.data_args.image_processor_aux_list
                video_file = os.path.join(self.data_args.image_folder, video_file)
                
                if "s3://" in video_file:
                    try:
                        video_bytes = self.client.get(dat["video"])
                        
                        vr = VideoReader(BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
                        sample_fps = round(
                                vr.get_avg_fps() / 1
                            )
                        frame_idx = [i for i in range(0, len(vr), sample_fps)]
                        image = vr.get_batch(frame_idx).asnumpy()
                        image_size = image[0].shape[:2]
                    except:
                        print("fail to load video: ", video_file, flush=True)
                        return self.__getitem__(19)
                elif os.path.exists(video_file):
                    try:
                        if video_file.endswith(".npy"):
                            image = np.load(video_file)
                            image_size = image[0].shape[:2]
                        elif video_file.endswith(".gif"):
                            video = Image.open(video_file)
                            image = []
                            for frame in ImageSequence.Iterator(video):
                                frame_copy = frame.copy()
                                image.append(frame_copy.convert("RGB"))
                            image_size = image[0].size
                        elif os.path.isdir(video_file):
                            files = [f for f in sorted(os.listdir(video_file))]
                            image = []
                            for file in files:
                                image.append(
                                    Image.open(os.path.join(video_file, file)).convert(
                                        "RGB"
                                    )
                                )
                            image_size = image[0].size
                        else:
                            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                            sample_fps = round(
                                vr.get_avg_fps() / self.data_args.video_fps
                            )
                            frame_idx = [i for i in range(0, len(vr), sample_fps)]
                            image = vr.get_batch(frame_idx).asnumpy()
                            image_size = image[0].shape[:2]
                        if self.data_args.uniform_sample:
                            image, sample_indices = uniform_sample(image, num_sample=200)

                    except:
                        print("fail to load video: ", video_file, flush=True)
                        return self.__getitem__(0)
                else:
                    print("Not exist: ", video_file, flush=True)
                    return self.__getitem__(0)
            
            # if not has_audio:
            image, sample_indices = uniform_sample(image, num_sample=224)
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result

            if self.data_args.image_aspect_ratio != "pad":
                raise NotImplementedError("Only pad is supported for now.")

            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                if not isinstance(image_aux, Image.Image):
                    frame_list = []
                    for frame in image_aux:
                        if not isinstance(frame, Image.Image):
                            frame = Image.fromarray(frame)
                        frame_aux = expand2square(
                            frame, tuple(int(x * 255) for x in processor_aux.image_mean)
                        ).resize((target_resolution, target_resolution))
                        frame_aux = processor_aux.preprocess(
                            frame_aux, return_tensors="pt"
                        )["pixel_values"][0]
                        frame_list.append(frame_aux)
                    image_aux = torch.stack(frame_list)
                else:
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                    image_aux = processor_aux.preprocess(
                        image_aux, return_tensors="pt"
                    )["pixel_values"][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)  # pyre-fixme
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], prompts=data_dict["prompts"][0]
            )
        if (data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict["image_aux_list"] = image_aux_list  # pyre-fixme
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            image_list = []
            for processor_aux in processor_aux_list:
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                image_list.append(
                    torch.zeros(
                        3,
                        target_resolution,
                        target_resolution,
                    )
                )
            data_dict["image_aux_list"] = image_list
            image_size = (crop_size, crop_size)
        data_dict["image_size"] = image_size  # pyre-fixme
        data_dict["video_indices"] = sample_indices
        
        has_audio = self._has_audio(dat)
        if has_audio:
            try:
                audio_file = dat["audio"]
                audio_file = os.path.join(self.data_args.image_folder, audio_file)
                audio_data = {"audio": [{'audio_file': audio_file,
                            'start_time': None,
                            'end_time': None}]
                            }
                if hasattr(self.data_args, "audio_processor"):
                    audio_processor = self.data_args.audio_processor
                    audio = audio_processor(audio_data)
                else:
                    audio = None
                data_dict["audio"] = audio  # pyre-fixme
            except Exception as e:
                print(e)
                print("fail to load audio: ", dat["audio"], flush=True)
                data_dict["audio"] = None  # pyre-fixme
        else:
            data_dict["audio"] = None  # pyre-fixme
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list  # pyre-fixme
    image_position: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  # pyre-fixme

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        video_indices = [instance["video_indices"] for instance in instances]
        prompts = [instance["prompts"] for instance in instances]
        audios = [instance["audio"] for instance in instances]
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)

        if padding_side == "left":
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (max_length - t.shape[0], 0),
                        "constant",
                        self.tokenizer.pad_token_id,
                    )
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (max_length - t.shape[0], 0), "constant", IGNORE_INDEX
                    )
                )
                for t in labels
            ]
        else:
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (0, max_length - t.shape[0]),
                        "constant",
                        self.tokenizer.pad_token_id,
                    )
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (0, max_length - t.shape[0]), "constant", IGNORE_INDEX
                    )
                )
                for t in labels
            ]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # pyre-fixme
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position + 1 :] = input_ids[
                    i, image_position:-1
                ]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position + 1 :] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position + 1 :] = attention_mask[
                    i, image_position:-1
                ]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
        image_sizes = [instance["image_size"] for instance in instances]
        (
            new_input_ids,
            new_labels,
            new_attention_mask,
            new_position_ids,
            im_aux_attention_masks_list,
        ) = prepare_multimodal_data(
            input_ids,
            labels,
            attention_mask,
            image_sizes,
            image_token_len,
            image_aux_token_len_list,
            max_length,
        )
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list,
            video_indices=video_indices,
            prompts=prompts,
            audios=audios
        )
        batch["image_sizes"] = image_sizes
        if "image_aux_list" in instances[0]:
            image_aux_list = [instance["image_aux_list"] for instance in instances]
            image_aux_list = [
                list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)
            ]
            if all(
                x is not None and x.shape == image_aux_list[0][0].shape
                for x in image_aux_list[0]
            ):
                batch["images"] = [
                    torch.stack(image_aux) for image_aux in image_aux_list
                ]
            else:
                batch["images"] = image_aux_list

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args  # pyre-fixme
) -> Dict:  # pyre-fixme
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator_kwargs = {
        "tokenizer": tokenizer,
    }

    if hasattr(data_args, "image_token_len"):
        data_collator_kwargs["image_token_len"] = data_args.image_token_len

    if hasattr(data_args, "vision_tower_aux_token_len_list"):
        data_collator_kwargs["image_aux_token_len_list"] = (
            data_args.vision_tower_aux_token_len_list
        )
    else:
        data_collator_kwargs["image_aux_token_len_list"] = [data_args.image_token_len]

    if hasattr(data_args, "image_position"):
        data_collator_kwargs["image_position"] = data_args.image_position

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)  # pyre-fixme

    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    # dist.init_process_group(backend="gloo", timeout=datetime.timedelta(hours=8))
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global_rank = get_global_rank()
    local_rank = get_local_rank()

    torch.distributed.barrier()

    # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
    training_args.output_dir = model_args.output_model_filename
    # pyre-fixme[16]: `DataClass` has no attribute `local_dir`.
    model_args.local_dir = model_args.output_model_filename

    bnb_model_from_pretrained_args = {}

    # pyre-fixme[16]: `DataClass` has no attribute `vision_tower`.
    if model_args.vision_tower_aux_list is not None:
        if "qwen" in model_args.input_model_filename.lower():
            model = CambrianQwenForCausalLM.from_pretrained(  # pyre-fixme
                model_args.input_model_filename,  # pyre-fixme
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),  # pyre-fixme
                **bnb_model_from_pretrained_args,
            )
        elif "llama" in model_args.input_model_filename.lower():
            # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
            #  `from_pretrained`.
            model = CambrianLlamaForCausalLM.from_pretrained(
                # pyre-fixme[16]: `DataClass` has no attribute `input_model_local_path`.
                model_args.input_model_filename,
                **bnb_model_from_pretrained_args,
            )
        else:
            raise NotImplementedError(
                f"{model_args.model_name_or_path} is not supported yet"
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.input_model_filename,
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    # pyre-fixme[16]: `DataClass` has no attribute `freeze_backbone`.
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # pyre-fixme[16]: `DataClass` has no attribute `gradient_checkpointing`.
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        non_lora_params = []
        lora_trainable_params = []
        
        non_lora_trainable_params = [
        "model.frame_seg",
        "model.vision_sampler_0.layers.0.pos_embed_0",
        "model.vision_sampler_0.layers.0.pos_embed_1",
        "model.vision_sampler_0.layers.1.pos_embed_0",
        "model.vision_sampler_0.layers.1.pos_embed_1",
        "model.vision_sampler_0.layers.2.pos_embed_0",
        "model.vision_sampler_0.layers.2.pos_embed_1",
        "Qformer"]
        
                
        for name, param in model.named_parameters():
            for listed_name in non_lora_trainable_params:
                if listed_name in name:
                    param.requires_grad = True
                    # print(name)

        for name, p in model.named_parameters():
            if not "lora" in name:
                non_lora_params.append(name)
            else:
                lora_trainable_params.append(name)
        # for name, p in model.named_parameters():
        #     if p.requires_grad and name in non_lora_params:
        #         print(name)
        model.config._attn_implementation_autoset = False
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        # pyre-fixme[16]: `DataClass` has no attribute `model_max_length`.
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # pyre-fixme[16]: `DataClass` has no attribute `version`.
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif model_args.version == "phi3":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif (
        model_args.version == "llama3"
        or model_args.version == "llama3_1"
        or model_args.version == "llama3_2"
    ):
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif model_args.version == "qwen":
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        if tokenizer.pad_token is None:
            print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_list`.
    if model_args.vision_tower_aux_list is not None:
        # pyre-fixme[16]: `DataClass` has no attribute `unfreeze_mm_vision_tower`.
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_token_len_list`.
        model_args.vision_tower_aux_token_len_list = json.loads(
            model_args.vision_tower_aux_token_len_list
        )
        # pyre-fixme[16]: `DataClass` has no attribute `query_num_list`.
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None,  # FSDP or not, flag should be the same as None to avoid creation error
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    if training_args.bf16:                    
                        vision_tower_aux.to(
                            dtype=torch.bfloat16, device=training_args.device  # pyre-fixme
                        )
                    if training_args.fp16:
                        vision_tower_aux.to(
                            dtype=torch.float16, device=training_args.device  # change to float16
                        )
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [  # pyre-fixme
                vision_tower_aux.image_processor
                for vision_tower_aux in vision_tower_aux_list
            ]
        data_args.is_multimodal = True  # pyre-fixme

        # our config
        model.config.pretrained_qformer = model_args.pretrained_qformer
        model.config.audio_input = model_args.audio_input
        model.config.unfreeze_mm_compressor = model_args.unfreeze_mm_compressor
        model.config.unfreeze_audio_encoder = model_args.unfreeze_audio_encoder
        model.config.max_num_segments = model_args.max_num_segments
        model.config.text_input = model_args.text_input
        model.config.query_type = model_args.query_type
        model.config.context_token_num = model_args.context_token_num
        model.config.add_static = model_args.add_static
        
        model.config.image_aspect_ratio = data_args.image_aspect_ratio  # pyre-fixme
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position  # pyre-fixme
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token  # pyre-fixme

        # pyre-fixme
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            # tune_modules = [
            #     "query_tokens",
            #     "Qformer",
            #     "vision_proj",
            #     "text_proj",
            #     "qformer_proj",
            #     "frame_seg"
            # ]
            tune_modules = [
                "mm_projector",
                "pos_emb",
                "vision_sampler",
                "vision_sampler_layers",
                "vision_query",
                "image_newline",
            ]
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter  # pyre-fixme
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.image_token_len = model_args.image_token_len = (  # pyre-fixme
            model_args.image_token_len
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr  # pyre-fixme
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr  # pyre-fixme
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr  # pyre-fixme
        training_args.use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = (
            data_args.vision_tower_aux_token_len_list
        ) = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = model_args.image_token_len
        model.config.is_st_sampler = model_args.is_st_sampler  # pyre-fixme
        data_args.image_token_len = model_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # initialize compression modules    
    # from IPython import embed
    # embed()
    unfreeze_mm_compressor = model_args.unfreeze_mm_compressor
    if model_args.pretrained_qformer:
        model.get_model().initialize_compressor(model.config, model_args.pretrained_qformer, model_args.context_token_num)
    
    # for p in model.model.Qformer.cls.parameters():
    #     p.requires_grad = False
                
    audio_input = model_args.audio_input
    if audio_input:
        print("Initializing audio modules...")
        model.get_model().initialize_audio(None)
        audio_encoder = model.get_model().audio_encoder
        
        data_args.audio_processor = Processor("./checkpoints/audio_encoder/whisper-large-v3")
        audio_encoder.to(
                    dtype=torch.float16, device=training_args.device  # pyre-fixme
                )
        
        unfreeze_audio_encoder = model_args.unfreeze_audio_encoder
        if unfreeze_audio_encoder:
            for p in audio_encoder.parameters():
                p.requires_grad = True
            
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(
        p.numel() for p in model.get_model().parameters() if p.requires_grad
    )
    print(f"Totol params: {total_params / 1024 / 1024}M")
    print(f"Trainable params: {trainable_params / 1024 / 1024}M")
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    # pyre-fixme
    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)

    os.environ[f"FSDP_USE_ORIG_PARAMS"] = "true"
    # pyre-fixme[16]: `DataClass` has no attribute `fsdp_config`.
    training_args.fsdp_config["use_orig_params"] = True
    # training_args.fsdp_config["sync_module_states"] = True
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    callbacks = []
    # configure TensorboardCallback to upload to manifold
    callbacks.append(
        TensorBoardCallback(
            SummaryWriter(
                log_dir=os.path.join(
                    # pyre-fixme[16]: `DataClass` has no attribute
                    #  `output_model_filename`.
                    model_args.output_model_filename,
                    TENSORBOARD_LOG_DIR_NAME,
                ),
                comment="",
                purge_step=None,
                max_queue=10,
                flush_secs=120,
                filename_suffix=str(uuid.uuid4()),
            )
        )
    )

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.args.save_safetensors = False
    # trainer.args.save_only_model = True
    # trainer.args.rank0_only=True
    # pyre-fixme[16]: `DataClass` has no attribute `output_dir`.
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # pyre-fixme[16]: `LLaVATrainer` has no attribute `train`.
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # pyre-fixme[16]: `LLaVATrainer` has no attribute `save_state`.
    trainer.save_state()

    if training_args.lora_enable:
        from tdc.mm_trainer import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(
        trainer=trainer,
        # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
        output_dir=model_args.output_model_filename,
    )


if __name__ == "__main__":
    train()
