import json
import os
import random
from PIL import Image

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

# Function to clean dataset entries
def clean_dataset(data):
    cleaned_data = []
    for entry in data:
        # Load the image using PIL
        image = Image.open(entry["image_path"])
        cleaned_entry = {
            "image": image,  # Replace "image_path" with "image" and load the image
            "ground_truth": entry["response"]  # Replace "response" with "ground_truth"
        }
        cleaned_data.append(cleaned_entry)
    return cleaned_data

# # Save each split to separate JSON files with prompt
# with open(os.path.join('training_data', level, 'train_dataset.json'), 'w') as train_file:
#     json.dump(flatten_data(train_data), train_file, indent=4)

# with open(os.path.join('training_data', level, 'val_dataset.json'), 'w') as val_file:
#     json.dump(flatten_data(val_data), val_file, indent=4)

# with open(os.path.join('training_data', level, 'test_dataset.json'), 'w') as test_file:
#     json.dump(flatten_data(test_data), test_file, indent=4)

# Clean datasets
cleaned_train_data = clean_dataset(train_data)
cleaned_val_data = clean_dataset(val_data)
cleaned_test_data = clean_dataset(test_data)

# Save cleaned datasets
with open('training_data/train_dataset.json', 'w') as train_cleaned_file:
    json.dump(cleaned_train_data, train_cleaned_file)

with open('training_data/val_dataset.json', 'w') as val_cleaned_file:
    json.dump(cleaned_val_data, val_cleaned_file)

with open('training_data/test_dataset.json', 'w') as test_cleaned_file:
    json.dump(cleaned_test_data, test_cleaned_file)


#####
# Run this I gotta go...
#####