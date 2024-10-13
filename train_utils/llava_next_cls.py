import lightning as L
import torch
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
from transformers import AutoProcessor
from datasets import load_dataset, load_from_disk
import os
import json  # Add this import

"""
Setup - collate functions
"""
# Load configuration from JSON file
with open(os.path.join('train_utils', 'train_config.json'), 'r') as f:
    train_config = json.load(f)

LEVEL = train_config.get("LEVEL") 
MODEL_ID = train_config.get("MODEL_ID") 
MAX_LENGTH = train_config.get("MAX_LENGTH")
NUM_WORKERS = train_config.get("WORKERS") #0

print("Loading the dataset")
hf_dataset = load_from_disk(os.path.join('training_data', f'hf_dataset_{LEVEL}'))
train_dataset = hf_dataset['train']
val_dataset = hf_dataset['validation']

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right


def train_collate_fn(examples):
    images = []
    texts = []
    for example in examples:
        image, image_id, prompt_str, ground_truth = example
        images.append(example[image])

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example[prompt_str]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example[ground_truth]},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(message)
        texts.append(text_prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, image_sizes, labels

def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    for example in examples:
        image, image_id, prompt_str, ground_truth = example
        images.append(image)

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": example[prompt_str]},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(message)
        texts.append(text_prompt)
        answers.append(example[ground_truth])
        

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    image_sizes = batch["image_sizes"]

    return input_ids, attention_mask, pixel_values, image_sizes, answers


class LlavaNextModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, image_sizes, labels = batch

        outputs = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_sizes=image_sizes,
                            labels=labels
                          )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)
