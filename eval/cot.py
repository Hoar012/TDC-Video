import torch
import numpy as np
from tdc.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tdc.conversation import conv_templates, SeparatorStyle
from tdc.mm_datautils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)

def LVCoT(model, video, image_sizes, tokenizer, version, question, max_forward=2):
    outputs = []
    
    for i in range(max_forward):
        conv = conv_templates[version].copy()
        conv.tokenizer = tokenizer
        
        seg_len = video[0].shape[1] // max_forward
        sub_video = []
        for v_feature in video:
            sub_video.append(v_feature[:,i*seg_len : (i+1)*seg_len])
        if sub_video[0].shape[1] <= 0:
            print("no video")
            continue

        # prefix_prompt = f"Summarize key information from the video segment between {start_time}s and {end_time}s that is relevant to answering the question: {question}"
        prefix_prompt = f"Describe the video segment in detail."
        prompt = "<image>\n" + prefix_prompt

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=sub_video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=128,
                use_cache=False,
                stopping_criteria=[stopping_criteria],
                prompt = prefix_prompt
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        
        outputs.append(f"{pred}")
    return outputs
