from datasets import load_from_disk
import os

hf_dataset = load_from_disk(os.path.join('training_data', 'hf_dataset_sample'))
train_dataset = load_from_disk(os.path.join('training_data', 'hf_dataset_sample'), split='train')
val_dataset = load_from_disk(os.path.join('training_data', 'hf_dataset_sample'), split='validation')

print(train_dataset)