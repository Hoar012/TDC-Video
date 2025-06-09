

torchrun --nproc_per_node=3 ./eval/eval_musicQA.py --model_path "./checkpoints/mm_cambrian_llama3_2_3-2-av" --model_name cambrian_llama --version llama3 --data_path /data/haohr/AV_data/Music-AVQA --test_file /data/haohr/AV_data/Music-AVQA/avqa_test-2-27.json