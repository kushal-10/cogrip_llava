from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from datasets import load_from_disk
import os
from tqdm import tqdm

LEVEL = "easy"

MAX_LENGTH = 384
REPO_ID = f"Koshti10/llava-1.5-7b-ft-{LEVEL}"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
test_dataset = hf_dataset['test']

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
# Load the base model with adapters on top
model = LlavaForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
)

count = 0

for i in tqdm(range(1000), desc="Evaluating"):
    example = test_dataset[i]
    test_image = example['image']

    prompt = f"USER: <image>\n{example['prompt']}\nASSISTANT:"
    inputs = processor(text=prompt, images=[test_image], return_tensors="pt").to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

    # Decode back into text
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Extract move
    move = generated_texts[0].split('ASSISTANT:')[-1].strip()
   

    # print("Generated Text:")
    # print(generated_texts)
    # print("GT:")
    # print(example['ground_truth'])

    if example['ground_truth'] == move:
        count += 1
    

print(count/len(test_dataset))