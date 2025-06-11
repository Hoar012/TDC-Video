import json
import ast
import numpy as np
with open("avsd_scores.json") as f:
    outputs = json.load(f)

scores = []
for output in outputs:
    
    try:
        d = ast.literal_eval(output)
        score = d["score"]
        scores.append(score)
    except (ValueError, SyntaxError) as e:
        print(output)
    
print("Average score:", np.mean(scores))