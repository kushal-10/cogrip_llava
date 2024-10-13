from datasets import load_from_disk
import os
import logging
import base64
import requests
import pandas as pd
from tqdm import tqdm

logging.basicConfig(filename=os.path.join('results', 'evaluation.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_gpt(LEVEL='easy'):
    """
    Function to evaluate GPT models
    """

    hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
    test_dataset = hf_dataset['test']

    predictions = []
    gts = []
    image_ids = []
    for i in tqdm(range(len(test_dataset)), desc='Evaluating GPT 4o'):
        example = test_dataset[i]
        image, image_path, prompt, gt = example
        # Strip quotes from the image_path
        cleaned_image_path = example[image_path].strip('"')
        # print(os.path.exists(cleaned_image_path))

        image_ids.append(cleaned_image_path)
        gts.append(example[gt])

        # Getting the base64 string
        base64_image = encode_image(cleaned_image_path)

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": example[prompt] + " Answer in one word only."
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 10
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        generated_text = response.json()['choices'][0]['message']['content']

        predictions.append(generated_text.lower())
        
    gpt_responses = {
        'predictions': predictions,
        'gts': gts,
        'ids': image_ids 
    }

    gpt_df = pd.DataFrame(gpt_responses)
    save_path = os.path.join('results', 'gpt.csv')
    gpt_df.to_csv(save_path, index=False)

if __name__ == '__main__':

    evaluate_gpt('sample')
