import os
import pandas as pd
import json



with open("additional_jsons/easy/train.json", "r") as f:
    train_data = json.load(f)
with open("additional_jsons/easy/val.json", "r") as f:
    val_data = json.load(f)

move_counter = {
    'right': 0,
    'left': 0,
    'up': 0,
    'down': 0
}

for i in range(len(train_data)):
    episode_data = train_data[i]
    for j in range(len(episode_data)):
        move = episode_data[j]["ground_truth"]
        if move in move_counter:
            move_counter[move] += 1

for i in range(len(val_data)):
    episode_data = val_data[i]
    for j in range(len(episode_data)):
        move = episode_data[j]["ground_truth"]
        if move in move_counter:
            move_counter[move] += 1

print(move_counter)
 