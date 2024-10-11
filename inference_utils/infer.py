from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
import torch
from datasets import load_from_disk
import os
from tqdm import tqdm
import argparse  # Add this import

# LEVEL = "easy"
# eval_base = True
# MODEL_ID = "llava-hf/llava-1.5-7b-hf"
# REPO_ID = f"Koshti10/llava-1.5-7b-ft-{LEVEL}"


def evaluate(level, model, type='raw'):
    """
    Function to evaluate the Finetuned and base models
    """
    MAX_LENGTH = 50
    MODEL_ID = model
    LEVEL = level

    hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
    test_dataset = hf_dataset['test']

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

    # Load the base model with adapters on top
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    if type == 'raw':
        count = 0
        steps = 500
        for i in tqdm(range(steps), desc="Evaluating"):
            example = test_dataset[i]
            test_image = example['image']

            prompt = f"USER: <image>\n{example['prompt']}. Answer in one word only.\nASSISTANT:"
            inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

            # Decode back into text
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Extract move
            move = generated_texts[0].split('ASSISTANT:')[-1].strip()
        
            print(f"Prideiction : {move.lower()} \n\n")
            print(f"GT : {example['ground_truth']} \n\n")

            if example['ground_truth'] == move.lower():
                count += 1
            
            # print(count/(i+1))
            # print(count, i+1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Finetuned and base models")
    parser.add_argument('--level', type=str, default='easy', help='Level of the dataset')
    parser.add_argument('--model', type=str, required=True, help='Model ID to evaluate, FT/Base HF repo')
    parser.add_argument('--type', type=str, default='raw', help='Type of evaluation (raw, episodic)')

    args = parser.parse_args()

    evaluate(args.level, args.model, args.type)

"""
python3 inference_utils/infer.py --level easy --model Koshti10/llava-1.5-7b-ft-pentomino-easy --type raw
"""
    