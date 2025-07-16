# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.


import datetime
import json
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
import time

import sys
sys.path.append('./')
import numpy as np
from PIL import Image

import torch

from tdc.builder import load_pretrained_model
from tdc.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tdc.conversation import conv_templates, SeparatorStyle
from tdc.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
from torch import distributed as dist
from tqdm import tqdm

from transformers.trainer_pt_utils import IterableDatasetShard

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path

        data_list = {
            "count": ("json/4_count.json", f"video/4_count", "video"),
            "ego": ("json/3_ego.json", f"video/3_ego", "video"),
            "needle": ("json/2_needle.json", f"video/2_needle", "video"),
            "order": ("json/5_order.json", f"video/5_order", "video"),
            "plotQA": ("json/1_plotQA.json", f"video/1_plotQA", "video"),
            "anomaly_reco": (
                "json/6_anomaly_reco.json",
                f"video/6_anomaly_reco",
                "video",
            ),
            "topic_reasoning": (
                "json/7_topic_reasoning.json",
                f"video/7_topic_reasoning",
                "video",
            ),
        }

        list_data_dict = []
        for k, v in data_list.items():
            with open(os.path.join(data_path, v[0]), "r") as f:
                json_data = json.load(f)
            for data in json_data:
                question, answer = self.qa_template(data)
                list_data_dict.append(
                    {
                        "task_type": k,
                        "video": os.path.join(self.data_path, v[1], data["video"]),
                        "video_name": data["video"],
                        "question": data["question"],
                        "prompt": question,
                        "answer": answer,
                    }
                )

        # pyre-fixme[4]: Attribute must be annotated.
        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question += (
            "Respond with only the letter (A, B, C or D) of the correct option.\n"
        )
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        return iter(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, i):
        return self.data[i]


def train(args) -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    
    version = args.version
    model_name = args.model_name
    model_base = args.model_base
    model_path = args.model_path

    torch.distributed.barrier()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,  # pyre-fixme
        model_base,
        model_name,
        device_map=None,
    )
    model.get_model().config.dino_threshold = 0.82
    model.get_model().config.drop_threshold = 0.77
    model.config.use_cache = True
    model.cuda()
    dataset = EvalDataset(
        data_path=args.data_path,
    )
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    output = []
    final_output = [None] * world_size

    for line in tqdm(shard_dataset):
        video_name = line["video_name"]
        answer = line["answer"]
        t_prompt = qs = line["prompt"]
        task_type = line["task_type"]
        video_path = line["video"]

        if os.path.exists(video_path):
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = round(vr.get_avg_fps())
            frame_idx = [
                    i
                    for i in range(0, len(vr), round(fps / 0.5))
                ]
            if len(frame_idx) > 1000:
                frame_idx = [
                    frame_idx[i]
                    for i in range(0, len(frame_idx), len(frame_idx) // 1000)
                ]
            video = vr.get_batch(frame_idx).asnumpy()
            image_sizes = [video[0].shape[:2]]
            video = process_images(video, image_processor, model.config)
            video = [item.unsqueeze(0) for item in video]
        else:
            video = np.zeros((1, 1024, 1024, 3)).astype(np.uint8)
            image_sizes = [(1024, 1024)]
            video = process_images(video, image_processor, model.config)

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        if "llama3" in version:
            input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=5,  
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                prompt=t_prompt
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        if pred.endswith(stop_str):
            pred = pred[: -len(stop_str)]
            pred = pred.strip()
        pred = pred.replace("Answer", "")

        letters = ["A", "B", "C", "D"]

        pred_answer = re.findall("[\(\ \[]*([A-D])[\)\.\ \]]*", pred)

        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip("()")
        if pred_answer in letters:
            pred_idx = letters.index(pred_answer)
            pred = letters[pred_idx]
        else:
            print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
            pred_idx = 2
            pred = letters[pred_idx]

        ans_id = uuid.uuid4()
        output.append(
            {
                "question": line["question"],
                "prompt": qs,
                "answer": answer,
                "pred": pred,
                "task_type": task_type,
                "answer_id": str(ans_id),
                "model_id": model_name,
                "video_name": video_name,
                "metadata": {},
            }
        )

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output))
    global_rank = dist.get_rank()
    if global_rank == 0:
        if not os.path.exists("./results/MLVU"):
            os.mkdir("./results/MLVU")
            
        save_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        
        with open(
            os.path.join("./results/MLVU", f"outputs-{save_time}.json"),
            "w",
        ) as f:
            json.dump(all_output, f)

        correct = 0
        total = 0
        acc_dict = {}
        for output in all_output:
            pred = output["pred"]
            gt = output["answer"]
            task_type = output["task_type"]
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]
            acc_dict[task_type][1] += 1
            total += 1

            if pred == gt:
                acc_dict[task_type][0] += 1
                correct += 1

        final_res = dict()
        total = 0
        idx = 0
        for k, v in acc_dict.items():
            idx += 1
            final_res[k] = v[0] / v[1] * 100
            total += final_res[k]
        final_res["Acc"] = total / idx
        print(final_res, flush=True)
        
        acc_dict["model_path"] = model_path
        acc_dict["model_base"] = model_base
        with open(os.path.join("./results/MLVU", f"result-{save_time}.json"), "w") as f:
            json.dump(acc_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/TDC-Qwen2-7B")
    parser.add_argument('--model_base', default=None)
    parser.add_argument('--model_name', default="cambrian_qwen")
    parser.add_argument('--version', default="qwen")
    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()
        
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    
    train(args)
