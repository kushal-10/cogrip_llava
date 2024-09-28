# cogrip_llava

## Overview
The `cogrip_llava` repository is designed for the Individual Module.

#### Directory Structure
Here is the general directory structure of the `cogrip_llava` repository:

cogrip_llava/
│
├── dataset/
│ ├── create_boards.py
│ └── other_dataset_files/
│
├── grip_env/
│ ├── init.py
│ ├── environment.py
│ ├── layout.py
│ └── pieces.py
│
├── tests/
│ ├── test_create_boards.py
│ └── other_tests/
│
├── README.md
└── requirements.txt

#### Initial Setup
**Clone the Repository**:
   ```bash
   git clone https://github.com/kushal-10/cogrip_llava.git
   cd cogrip_llava
   ```

**Install Dependencies**:
   Ensure you have the required libraries installed. You can do this using pip:
   ```bash
   pip install -r requirements.txt
   ```

##### 1) Create Boards
The `dataset/create_boards.py` script is responsible for generating data boards from raw dataset inputs. It processes the input data, organizes it into a structured format, and saves the output for further analysis or training purposes.

**Run the Script**:
   Execute the script with the necessary parameters:
   ```bash
   python dataset/create_boards.py --board_size <size> --num_pieces <number> --shapes <shapes> --colours <colours> --num_boards <number> --level <difficulty> --path <output_path>
   ```

**Parameters**:
- `--board_size`: Size of the board (default: 18).
- `--num_pieces`: Number of pieces to use (default: 4).
- `--shapes`: Space-separated string of shapes (default: "P T X Z U W").
- `--colours`: Space-separated string of colors (default: "red blue green yellow magenta cyan").
- `--num_boards`: Total number of boards to create for each combination of piece and color (default: 100).
- `--level`: Difficulty level (default: 'easy').
- `--path`: Output directory to save the generated boards (default: 'data').

**Example**:
   ```bash
   python dataset/create_boards.py --board_size 18 --num_pieces 4 --shapes "P T X Z U W" --colours "red blue green yellow magenta cyan" --num_boards 100 --level 'easy' --path 'data'
   ```

##### 2) Generate Paths from `agent_start_pos` to `target_pos`
Set variables under `dataset/generate_paths`, then run

```bash
python3 dataset/generate_paths.py
```

This will create a `data/level/metadata_path.json` that saves an additional field/key of `path` in the `metadata.json` file


##### 3) Now with the paths generated, create the whole training/finetuning dataset that will be used for the llava models

Run the follwing script with additional arguments, if required( level, size) from step1 - Create boards. This will create a folder `training_data/level/boards` and a meta data file that contains `image_path`, `prompt` and `response` for each instance under `training_data/level/training_metadata.json`

```bash
python3 dataset/generate_episodes.py
```

#### Finetuning
