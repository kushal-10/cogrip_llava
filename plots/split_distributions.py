import json
import numpy as np
import matplotlib.pyplot as plt

def plot_overlay_data(train_data, test_data, title, xlabel, ylabel):
    labels = train_data.keys()
    train_values = train_data.values()
    test_values = test_data.values()

    x = range(len(labels))  # X-axis positions for the bars

    plt.bar(x, train_values, width=0.4, label="Train", align="center")
    plt.bar([i + 0.4 for i in x], test_values, width=0.4, label="Test", align="center")
    
    plt.xticks([i + 0.2 for i in x], labels)  # Set labels in the center of both bars
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'plots/{xlabel.lower()}_{ylabel.lower()}.png', bbox_inches='tight')  # Save the plot locally


with open('data/easy/train.json', 'r') as f:
    train_data = json.load(f)

with open('data/easy/val.json', 'r') as f:  
    val_data = json.load(f)

with open('data/easy/test.json', 'r') as f: 
    test_data = json.load(f)

train_data_shapes = {}
train_data_colours = {}
train_data_positions = np.zeros((18, 18))

test_data_shapes = {}
test_data_colours = {}
test_data_positions = np.zeros((18, 18))

train_count = 0
val_count = 0
test_count = 0
for i in range(len(train_data)):
    data_obj = train_data[i]
    for j in range(len(data_obj)):
        data_instance = data_obj[j]
        if "info" in data_instance:
            train_count += len(data_instance["path"])
            target_gridpoints = data_instance["info"][0]["piece_grids"]
            target_shape = data_instance["info"][0]["piece_shape"]
            target_colour = data_instance["info"][0]["piece_colour"]

            if target_shape not in train_data_shapes:
                train_data_shapes[target_shape] = 0
            train_data_shapes[target_shape] += 1

            if target_colour not in train_data_colours:
                train_data_colours[target_colour] = 0
            train_data_colours[target_colour] += 1

            for gridpoint in target_gridpoints:
                train_data_positions[gridpoint[0], gridpoint[1]] += 1

for i in range(len(val_data)):
    data_obj = val_data[i]
    for j in range(len(data_obj)):
        data_instance = data_obj[j]
        if "info" in data_instance:
            val_count += len(data_instance["path"])
            target_gridpoints = data_instance["info"][0]["piece_grids"]
            target_shape = data_instance["info"][0]["piece_shape"]
            target_colour = data_instance["info"][0]["piece_colour"]

            if target_shape not in train_data_shapes:
                train_data_shapes[target_shape] = 0
            train_data_shapes[target_shape] += 1

            if target_colour not in train_data_colours:
                train_data_colours[target_colour] = 0
            train_data_colours[target_colour] += 1

            for gridpoint in target_gridpoints:
                train_data_positions[gridpoint[0], gridpoint[1]] += 1

for i in range(len(test_data)):
    data_obj = test_data[i]
    for j in range(len(data_obj)):
        data_instance = data_obj[j]
        if "info" in data_instance:
            test_count += len(data_instance["path"])
            target_gridpoints = data_instance["info"][0]["piece_grids"]
            target_shape = data_instance["info"][0]["piece_shape"]
            target_colour = data_instance["info"][0]["piece_colour"]    

            if target_shape not in test_data_shapes:
                test_data_shapes[target_shape] = 0
            test_data_shapes[target_shape] += 1

            if target_colour not in test_data_colours:
                test_data_colours[target_colour] = 0
            test_data_colours[target_colour] += 1   

            for gridpoint in target_gridpoints:
                test_data_positions[gridpoint[0], gridpoint[1]] += 1

plt.figure(figsize=(9, 9))
plt.imshow(train_data_positions, cmap='cool', interpolation='bilinear', extent=(1, 18, 18, 1))  # Create the heatmap
plt.colorbar()  # Add a colorbar to indicate the scale
plt.title('Heatmap showing the layout of target pieces in the training+val set')  # Add a title
plt.xlabel('X Position')  # Label for x-axis
plt.ylabel('Y Position')  # Label for y-axis
plt.savefig(f'plots/train_split.png', bbox_inches='tight')  # Save the plot locally
plt.close()


plt.figure(figsize=(9, 9))
plt.imshow(test_data_positions, cmap='cool', interpolation='bilinear', extent=(1, 18, 18, 1))  # Create the heatmap
plt.colorbar()  # Add a colorbar to indicate the scale
plt.title('Heatmap showing the layout of target pieces in the test set')  # Add a title
plt.xlabel('X Position')  # Label for x-axis
plt.ylabel('Y Position')  # Label for y-axis
plt.savefig(f'plots/test_split.png', bbox_inches='tight')  # Save the plot locally
plt.close()

plot_overlay_data(train_data_shapes, test_data_shapes, "Distribution of target shapes in the training and test sets", "Shape", "Count")
plt.close()

plot_overlay_data(train_data_colours, test_data_colours, "Distribution of target colours in the training and test sets", "Colour", "Count")
plt.close()

print(f"Train instances: {train_count}, Val instances: {val_count}, Test instances: {test_count}")
print(f"Train episodes: {len(train_data)}, Val episodes: {len(val_data)}, Test episodes: {len(test_data)}")