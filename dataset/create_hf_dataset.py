import json
import os
import random
from datasets import Dataset, DatasetDict, Features, Value, Image as HfImage
import argparse

def gen_hf_data(level="easy"):

    metadata_path = os.path.join('training_data', level, 'training_metadata.json')

    # Load the metadata from JSON
    with open(metadata_path, 'r') as f:
        json_data = json.load(f)

    # Shuffle and split data into training, validation, and test sets
    random.shuffle(json_data)  # Shuffle the data for randomness
    train_size = int(len(json_data) * 0.7)  # 70% for training
    val_size = int(len(json_data) * 0.15)    # 15% for validation
    test_size = len(json_data) - train_size - val_size  # Remaining 15% for testing

    # Convert list of lists and save the datasets as JSON files
    os.makedirs(os.path.join('training_data', f'hf_metadata_{level}'), exist_ok=True)
    
    # Save train data
    train_data = json_data[:train_size]
    with open(os.path.join('training_data', f'hf_metadata_{level}', 'train.json'), 'w') as f:
        json.dump(train_data, f)

    # Save validation data
    val_data = json_data[train_size:train_size + val_size]
    with open(os.path.join('training_data', f'hf_metadata_{level}', 'val.json'), 'w') as f:
        json.dump(val_data, f)

    # Save test data
    test_data = json_data[train_size + val_size:]
    with open(os.path.join('training_data', f'hf_metadata_{level}', 'test.json'), 'w') as f:
        json.dump(test_data, f)

    # Flatten the data for further processing
    # Remove the additional metadata for the HF dataset
    train_data = [item for sublist in train_data for item in sublist if "info" not in sublist]
    val_data = [item for sublist in val_data for item in sublist if "info" not in sublist]
    test_data = [item for sublist in test_data for item in sublist if "info" not in sublist]

    # Debugging: Print the sizes of each split
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    # Define dataset features
    features = Features({
        'image': HfImage(),  # Keep the original HfImage
        'image_string': Value(dtype='string'),  # Add a new field for the image as a string
        'prompt': Value(dtype='string'),
        'ground_truth': Value(dtype='string'),
    })

    # Create datasets using from_dict
    train_dataset = Dataset.from_dict({
        'image': [item['image'] for item in train_data],  # Keep the original image
        'image_string': [json.dumps(item['image']) for item in train_data],  # Convert image to string
        'prompt': [item['prompt'] for item in train_data],
        'ground_truth': [item['ground_truth'] for item in train_data],
    }, features=features)

    val_dataset = Dataset.from_dict({
        'image': [item['image'] for item in val_data],  # Keep the original image
        'image_string': [json.dumps(item['image']) for item in val_data],  # Convert image to string
        'prompt': [item['prompt'] for item in val_data],
        'ground_truth': [item['ground_truth'] for item in val_data],
    }, features=features)

    test_dataset = Dataset.from_dict({
        'image': [item['image'] for item in test_data],  # Keep the original image
        'image_string': [json.dumps(item['image']) for item in test_data],  # Convert image to string
        'prompt': [item['prompt'] for item in test_data],
        'ground_truth': [item['ground_truth'] for item in test_data],
    }, features=features)

    # Create DatasetDict for easy handling of splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset,
    })

    # Save the dataset
    dataset_dict.save_to_disk(os.path.join('training_data', f'hf_dataset_{level}'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--level', type=str, default='sample', help='Difficulty level - String - "sample", "easy", "medium", "hard"')
    
    args = parser.parse_args()
    gen_hf_data(level=args.level)
