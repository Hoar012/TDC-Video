import numpy as np
import torch
from tdc.builder import load_pretrained_model
from tdc.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tdc.conversation import conv_templates, SeparatorStyle
from tdc.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
from utils.processor import Processor

tokenizer, model, image_processor, context_len = load_pretrained_model(
    "checkpoints/TDC-Qwen2-7B", None, "cambrian_qwen",
)
audio_processor = Processor("checkpoints/audio_encoder/whisper-large-v3")

model.eval()
model.cuda()
video_path = "./examples/video1.mp4"
audio_path = "./examples/audio1.wav"
instruction = qs = "Describe this video in detail, what can you see and hear?"

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = float(vr.get_avg_fps())
frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
video = []
for frame_index in frame_indices:
    img = vr[frame_index].asnumpy()
    video.append(img)
video = np.stack(video)
image_sizes = [video[0].shape[:2]]
video = process_images(video, image_processor, model.config)
video = [item.unsqueeze(0) for item in video]

if audio_path is not None:
    audio_data = {
        "audio": [{'audio_file': audio_path, 'start_time': None, 'end_time': None}]
                }
    audio = audio_processor(audio_data)
else:
    audio = None
    
qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["qwen"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=video,
        image_sizes=image_sizes,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=128,
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        prompt=instruction,
        audio=audio
    )
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(pred)