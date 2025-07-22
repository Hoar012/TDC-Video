# Evaluation on General Video Understanding

## MVBench

Download [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) to `./eval/MVBench`

```bash
torchrun --nproc_per_node=8 ./eval/eval_mvbench.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/MVBench
```

## PerceptionTest

Download the valid split of [PerceptionTest](https://github.com/google-deepmind/perception_test) to `./eval/PerceptionTest`

```bash
torchrun --nproc_per_node=8 ./eval/eval_perception.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/PerceptionTest
```

## EgoSchema

Download [EgoSchema](https://github.com/egoschema/EgoSchema) to `./eval/EgoSchema`

```bash
torchrun --nproc_per_node=8 ./eval/eval_egoschema.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/EgoSchema
```

Then submit the result file `.csv` to [Kaggle](https://www.kaggle.com/competitions/egoschema-public/submissions)


## MLVU

Download [MVLU](https://huggingface.co/datasets/MLVU/MVLU) to `./eval/MLVU`

```bash
torchrun --nproc_per_node=8 ./eval/eval_mlvu.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/MLVU
```

Set the --use_lvcot flag to enable evaluation with LVCoT

```bash
torchrun --nproc_per_node=8 ./eval/eval_mlvu.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/MLVU --use_lvcot
```

## VideoMME

Download [VideoMME](https://github.com/BradyFU/Video-MME?tab=readme-ov-file#-dataset) to `./eval/VideoMME`

```
torchrun --nproc_per_node=8 ./eval/eval_videomme.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/VideoMME --use_subtitle
torchrun --nproc_per_node=8 ./eval/eval_videomme.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/VideoMME --use_subtitle --use_lvcot 
```

# Evaluation on Audio-Visual Comprehension

## Music-AVQA

Download the processed videos and audios from [Music-AVQA](https://huggingface.co/datasets/Hoar012/TDC_training_data) to `./eval/Music-AVQA`


```bash
torchrun --nproc_per_node=8 ./eval/eval_musicQA.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/Music-AVQA --test_file data/AV_data/Music-AVQA/avqa-test.json
```

## AVSD

Download the processed videos and audios from [AVSD](https://huggingface.co/datasets/Hoar012/TDC_training_data) to `./eval/AVSD`


```bash
torchrun --nproc_per_node=8 ./eval/eval_avsd.py --model_path Hoar012/TDC-Qwen2-7B --model_name cambrian_qwen --version qwen --data_path eval/AVSD --test_file eval/AVSD/avsd_val.jsonavqa-test.json
```