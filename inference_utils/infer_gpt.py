from datasets import load_from_disk
import os
import logging

logging.basicConfig(filename=os.path.join('results', 'evaluation.log'), level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_gpt(model_id, LEVEL='easy'):
    """
    Function to evaluate GPT models
    """

    hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
    test_dataset = hf_dataset['test']
