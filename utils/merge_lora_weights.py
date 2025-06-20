import sys, os
sys.path.append(os.getcwd())
import argparse
from tdc.builder import load_pretrained_model


def merge_lora(args):
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, "cambrian_qwen_lora", device_map='cpu'
    )

    model.save_pretrained(args.save_model_path, safe_serialization=False)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()
    merge_lora(args)