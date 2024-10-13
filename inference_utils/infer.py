from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
import torch
from datasets import load_from_disk
import os
from tqdm import tqdm
import argparse
import pandas as pd
import logging

logging.basicConfig(filename=os.path.join('results', 'evaluation.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate(level, model):
    """
    Function to evaluate the Finetuned and base models
    """
    MAX_LENGTH = 50
    MODEL_ID = model
    LEVEL = level

    hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
    test_dataset = hf_dataset['test']
    steps = len(test_dataset)

    if "-ft" in MODEL_ID:
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
    else:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
        model.to("cuda")


    if not os.path.exists('results'):
        os.makedirs('results')

    predictions = []
    ground_truths = []
    image_ids = []

    check_acc = 0
    for i in tqdm(range(steps), desc="Evaluating on test dataset"):
        example = test_dataset[i]
        test_image = example['image']
        image_id = example['image_string']

        if "-ft" in MODEL_ID:
            prompt = example['prompt']
        else:
            prompt = f"USER: <image>\n{example['prompt']}. Answer in one word only.\nASSISTANT:"

        inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

        # Decode back into text
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Extract move
        move = generated_texts[0].split('ASSISTANT:')[-1].strip()
    
        predictions.append(move)
        ground_truths.append(example['ground_truth'])
        image_ids.append(image_id)

        if move.lower() == example['ground_truth']:
            check_acc += 1

        if i % 50 == 0:
            logging.info(f'Accuracy at step {i}: {check_acc/(i+1)}')

    
    save_name =MODEL_ID.split('/')[-1] + ".csv"
    infer_data = {
        "predictions": predictions,
        "gts": ground_truths,
        "ids": image_ids
    }    
    infer_df = pd.DataFrame(infer_data)
    infer_df.to_csv(os.path.join('results', save_name), index=False)
    logging.info(f'Saved dataframe to {os.path.join("results", save_name)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Finetuned and base models")
    parser.add_argument('--level', type=str, default='easy', help='Level of the dataset')
    parser.add_argument('--model', type=str, required=True, help='Model ID to evaluate, FT/Base HF repo')

    args = parser.parse_args()

    evaluate(args.level, args.model)
