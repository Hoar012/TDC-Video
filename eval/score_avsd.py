import json
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "checkpoints/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("/mnt/petrelfs/haohaoran/code/tdc/results/AVSD/outputs-2025-05-16-12:18:13.json") as f:
    outputs = json.load(f)

scores = []
for output in tqdm(outputs):
    {"prompt": "<image>\nIs the person a man?", "pred": "No.", "correct_answer": "The person visable in the video is a woman, but It sounds as though there is a man having a conversation with her.", "answer_id": "8ead007f-e2e7-4d09-aa1b-d035d3a1a3a3", "model_id": "cambrian_qwen_lora", "video_name": "3MSZA.mp4"}
    question = output["prompt"].replace("<image>\n", "")
    correct_answer = output["correct_answer"]
    pred_answer = output["pred"]

    prompt = f"Please evaluate the following video-based question-answer pair: Question: {question}, Correct Answer: {correct_answer}, Predicted Answer: {pred_answer}. Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. For example, your response should look like this:" + "\{'pred': 'yes', 'score': 4\}."


    messages = [
        {"role": "system", "content": "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:------\#\#INSTRUCTIONS: - Focus on the meaningful match between the predicted answer and the correct answer. - Consider synonyms or paraphrases as valid matches. - Evaluate the correctness of the prediction compared to the answer. "},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    scores.append(response)

with open("avsd_scores.json", "w") as f:
    json.dump(scores, f, indent=4)