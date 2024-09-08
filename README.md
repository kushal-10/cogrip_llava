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

#### Usage
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



