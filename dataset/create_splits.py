import json
import os
import random

# Set variables
level = 'sample'

metadata_path = os.path.join('training_data', level, 'training_metadata.json')

with open(metadata_path, 'r') as f:
    json_data = json.load(f)

# Split data into training and test sets
random.shuffle(json_data)  # Shuffle the data for randomness
split_index = int(len(json_data) * 0.8)  # 80% for training, 20% for testing
train_data = json_data[:split_index]
test_data = json_data[split_index:]

# Save the splits to new JSON files
with open(os.path.join('training_data', level, 'train_data.json'), 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open(os.path.join('training_data', level, 'test_data.json'), 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

