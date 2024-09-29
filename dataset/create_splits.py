import json
import os
import random

# Set variables
level = 'sample'
metadata_path = os.path.join('training_data', level, 'training_metadata.json')

with open(metadata_path, 'r') as f:
    json_data = json.load(f)

# Split data into training, validation, and test sets
random.shuffle(json_data)  # Shuffle the data for randomness
train_size = int(len(json_data) * 0.7)  # 70% for training
val_size = int(len(json_data) * 0.15)   # 15% for validation
test_size = len(json_data) - train_size - val_size  # Remaining 15% for testing

train_data = json_data[:train_size]
val_data = json_data[train_size:train_size + val_size]
test_data = json_data[train_size + val_size:]

# Flatten the data
def flatten_data(data):
    return [item for sublist in data for item in sublist]

# Save each split to separate JSON files
with open(os.path.join('training_data', level, 'train_dataset.json'), 'w') as train_file:
    json.dump(flatten_data(train_data), train_file, indent=4)

with open(os.path.join('training_data', level, 'val_dataset.json'), 'w') as val_file:
    json.dump(flatten_data(val_data), val_file, indent=4)

with open(os.path.join('training_data', level, 'test_dataset.json'), 'w') as test_file:
    json.dump(flatten_data(test_data), test_file, indent=4)


