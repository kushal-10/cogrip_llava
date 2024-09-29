from datasets import load_dataset
import json


# Load datasets from flattened JSON files, Each json file has a single split (default - train)
train_dataset = load_dataset('json', data_files='training_data/sample/train_dataset.json', split='train')
val_dataset = load_dataset('json', data_files='training_data/sample/val_dataset.json', split='train')
test_dataset = load_dataset('json', data_files='training_data/sample/test_dataset.json', split='train')

# Optionally, you can print the dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

print(train_dataset)
