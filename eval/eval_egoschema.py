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
import pandas as pd

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

        # pyre-fixme[4]: Attribute must be annotated.
        self.data = json.load(open(
                        os.path.join(
                        args.data_path, "questions.json"
                    ), "r"))

    def __len__(self) -> int:
        return len(self.data)

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
    model.get_model().config.drop_threshold = 0.8
    model.config.use_cache = True
    # model.cuda()
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

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    for line in tqdm(shard_dataset):
        video_name = line["google_drive_id"]
        idx = line["q_uid"]
        question = line["question"]
        a0 = line["option 0"]
        a1 = line["option 1"]
        a2 = line["option 2"]
        a3 = line["option 3"]
        a4 = line["option 4"]
        t_prompt = qs = f"Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nRespond with only the letter (A, B, C, D or E) of the correct option."
        video_path = os.path.join(
            args.data_path,
            "videos",
            f"{idx}.mp4",
        )
        for fmt in video_formats:
            temp_path = os.path.join(
                args.data_path,
                "videos",
                f"{idx}{fmt}",
            )
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        if os.path.exists(video_path):
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = round(vr.get_avg_fps())
            frame_idx = [
                    i
                    for i in range(0, len(vr), round(fps / 1))
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
        conv.tokenizer = tokenizer
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

        letters = ["A", "B", "C", "D", "E"]
        pred_answer = re.findall("[\(\ ]*[A-E][\)\ ]*", pred)
        if len(pred_answer) <= 0:
            pred_answer = [""]
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
                "idx": idx,
                "prompt": qs,
                "pred": pred_idx,
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
        if not os.path.exists("./results/EgoSchema"):
            os.mkdir("./results/EgoSchema")

        save_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

        with open(
            os.path.join("./results/EgoSchema", f"outputs-{save_time}.json"),
            "w",
        ) as f:
            json.dump(all_output, f)

        answers = json.load(
                open(
                    os.path.join(
                        args.data_path, "subset_answers.json"
                    ),
                    "r",
                )
            )

        correct = 0
        total = 0
        for output in all_output:
            if output["idx"] in answers:
                total += 1
                if int(output["pred"]) == int(answers[output["idx"]]):
                    correct += 1

        print(f"Accuracy: {correct / total}")
        result = {"acc": correct / total}
            
        with open(os.path.join("./results/EgoSchema", f"result-{save_time}.json"), "w") as f:
            json.dump(result, f)

        submission = []
        q_uids = []
        for output in all_output:
            item = {}
            item["q_uid"] = output["idx"]
            item["answer"] = int(output["pred"])
            if output["idx"] not in q_uids:
                q_uids.append(output["idx"])
                submission.append(item)
        df = pd.DataFrame(submission)
        df.to_csv(
            "./results/EgoSchema/submission.csv",
            index=False,
            columns=["q_uid", "answer"],
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/tdc_qwen")
    parser.add_argument('--model_base', default=None)
    parser.add_argument('--model_name', default="cambrian_qwen")
    parser.add_argument('--version', default="qwen")
    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)

    train(args)
