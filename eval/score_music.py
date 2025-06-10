import json
import numpy as np

results = json.load(open("/mnt/lustre/hanjiaming/petrelfs/code/tdc/results/MusicQA/outputs-2025-05-10-21:38:18.json", "r"))

scores = []
for res in results:
    
    if res["correct_answer"] in res["pred"].lower():
        scores.append(1)
    else:
        scores.append(0)
        print(res)
        
print(np.mean(scores))