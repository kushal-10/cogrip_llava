import json
import os
import random
from datasets import Dataset, DatasetDict, Features, Value, Image as HfImage

# Set variables
level = 'sample'
metadata_path = os.path.join('training_data', level, 'training_metadata.json')

# Load the metadata from JSON
with open(metadata_path, 'r') as f:
    json_data = json.load(f)

# Shuffle and split data into training, validation, and test sets
random.shuffle(json_data)  # Shuffle the data for randomness
train_size = int(len(json_data) * 0.7)  # 70% for training
val_size = int(len(json_data) * 0.15)    # 15% for validation
test_size = len(json_data) - train_size - val_size  # Remaining 15% for testing

# Convert list of lists
train_data = [item for sublist in json_data[:train_size] for item in sublist]
val_data = [item for sublist in json_data[train_size:train_size + val_size] for item in sublist]
test_data = [item for sublist in json_data[train_size + val_size:] for item in sublist]

# Debugging: Print the sizes of each split
print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

# Define dataset features
features = Features({
    'image': HfImage(),
    # 'prompt': Value(dtype='string'),
    'ground_truth': Value(dtype='string'),
})

# Create datasets using from_dict
train_dataset = Dataset.from_dict({
    'image': [item['image'] for item in train_data],
    # 'prompt': [item['prompt'] for item in train_data],
    'ground_truth': [item['ground_truth'] for item in train_data],
}, features=features)

val_dataset = Dataset.from_dict({
    'image': [item['image'] for item in val_data],
    # 'prompt': [item['prompt'] for item in val_data],
    'ground_truth': [item['ground_truth'] for item in val_data],
}, features=features)

test_dataset = Dataset.from_dict({
    'image': [item['image'] for item in test_data],
    # 'prompt': [item['prompt'] for item in test_data],
    'ground_truth': [item['ground_truth'] for item in test_data],
}, features=features)

# Create DatasetDict for easy handling of splits
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset,
})

# Save the dataset
dataset_dict.save_to_disk(os.path.join('training_data', 'hf_dataset_sample'))

