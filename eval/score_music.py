import json
import numpy as np

results = json.load(open("/mnt/petrelfs/haohaoran/code/LongVU/results/MusicQA/outputs-llama3_2_video_audio_300k.json", "r"))

scores = []
for res in results:
    
    if res["correct_answer"] in res["pred"].lower():
        scores.append(1)
    else:
        scores.append(0)
        print(res)
        
print(np.mean(scores))