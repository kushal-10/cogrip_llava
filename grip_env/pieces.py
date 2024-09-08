# Define the pentomino pieces and handle their rotations
# Output should be an array/list of 1x2 blocks to be marked in the pentomino grid environment

import numpy as np

# Piece definitions for 0 rotation
# Pieces are P, T, U, W, X, Z

# Shift from 5x5 to 3x3 grid
P = [
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 0]]

T = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0]]

U = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0]]

W = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 1]]

X = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]]

Z = [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1]]

# Create a dictionary item for the pieces
pieces_dict = {
    'P': P,
    'T': T,
    'U': U,
    'W': W,
    'X': X,
    'Z': Z
}

# Define a list of piece names from above dict - to select a piece randomly
PIECE_NAMES = ['P', 'T', 'U', 'W', 'X', 'Z']

# Define colours
# Define a dict of colours in RGB format- str: tuple

COLOURS = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'cyan': (0, 0, 255),
    'magenta': (255, 0, 255)
}

# Define a list of colour names from above dict - to select a colour randomly
COLOUR_NAMES = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

class PentominoPiece():
    '''
    Intialize a pentomino piece with a symbol, rotation and position
    Args:
        symbol: A single letter string correesponding to the piece shape
        rotation: An integer r in {0, 1, 2, 3} to define the angle of rotation corresponding to  r*pi/2
        position: The block in the Pentomino Board Grid coressponding to the center of the Piece Grid
    '''
    def __init__(self, symbol: str, rotation: int, position: np.array):
        self.symbol = symbol
        self.rotation = rotation
        self.position = [position[0]-1, position[1]-1] # Convert to 0-indexing - top-left corner of the piece
        
    def get_grid_locations(self) -> np.array:
        '''
        Get the locations of blocks to mark as a piece in the Pentomino Board Grid
        Returns:
            grid_marks: An np.array containing multiple vectors of length 2 for a single piece
                        (co-ordinates to be marked in the Pentomino Board Grid)
        '''
        
        default_piece_grid = pieces_dict[self.symbol]
        rotated_piece_grid = np.rot90(default_piece_grid, self.rotation)
     
        # Get the grid marks for the piece accoding to the Gym coordinate system = Inverted Y - axis
        grid_marks = []
        for i in range(rotated_piece_grid.shape[0]):
            for j in range(rotated_piece_grid.shape[1]):
                if rotated_piece_grid[i][j] == 1:
                    grid_marks.append(self.position + np.array([j, i]))
        grid_marks = np.array(grid_marks)

        return grid_marks

if __name__ == '__main__':
    piece1 = PentominoPiece('P', 0, np.array([1, 1]))
    print(piece1.get_grid_locations())