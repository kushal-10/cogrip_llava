import numpy as np

from grip_env.pieces import PentominoPiece, COLOURS, COLOUR_NAMES, PIECE_NAMES
from utils import layout_utils
import math

# Setting the layout fixed for now i.e start positions of each piece are at every 5x5 grid
class BoardLayout():
    '''
    This class is used to generate a random layout of pentomino pieces on a board.
    Args:
        board_size: The number of grids in one dimension ofthe Pentomino Board
        num_pieces: The number of pieces to be placed on the board, including the target piece
        shapes: A list of the pentomino shapes to be selected from
        colours: A list of the pentomino colours to be selected from
        seed: The seed for the random number generator
    '''
    def __init__(self, board_size: int, num_pieces: int, shapes: np.array, colours: np.array, seed: int):
        self.board_size = board_size 
        self.num_pieces = num_pieces
        self.shapes = shapes
        self.colours = colours
        self.mapped_regions = layout_utils.map_regions(self.board_size)
        np.random.seed(seed)
    
    def set_start_positions(self) -> np.array:
        '''
        Get start positions of everything on the board
        Args:
            regions: A list defining the regions where the piece will be spawned (['top', 'top left', 'right',...])
                    If None, then use all possible regions including the center grid
        Returns:
            all_start_positions - [[ax, ay], [p1x, p1y], [p2x, p2y], ....]
            The starting positions of agents and all the pieces (top left corner of 5x5 grid)
        '''

        # # Set the starting position at the center of the grid where the gripper will be spawned
        # center_sq = math.ceil((self.board_size)/2)
        # agent_start_pos = np.array([center_sq, center_sq], dtype=np.int64) # Get start position of agent
        # # Use this location to check for overlaps for new pieces generated
        # all_start_positions = np.array([agent_start_pos]) # Initialize with agent start position, so atleast one step is taken
        


        max_tries = 100  # Maximum number of tries
        tries = 0  # Counter for tries
        flag = True
        while flag:
            all_start_positions = [] # Initialize with empty list - Piece can be spawned on center gird as well, overlapping with agent

            # Select a random start position for each piece
            tries += 1  # Increment the try counter
            if tries > max_tries:  # Check if max tries exceeded
                print("Max tries exceeded - Restart Board Layout - Try increasing the board size or reducing the number of piecess")
                break  # Exit the main while loop

            spawn_choices = [[x, y] for x in range(self.board_size) for y in range(self.board_size)] # Get possible spawn locations across the board
            for i in range(self.num_pieces):
                random_choice = np.random.randint(0, len(spawn_choices)) # Select a random index
                piece_start_pos = spawn_choices[random_choice] # Random grid mark in the specified region

                # Draw randomly, until a valid value is found
                # This ensures no overlaps between pieces and center grid (central 3x3 will always be empty) 
                while not layout_utils.valid(self.board_size, piece_start_pos, all_start_positions):
                    # Remove invalid starting position and select a start position again
                    spawn_choices.remove(piece_start_pos) 
                    if not spawn_choices:  # Check if all positions are exhausted
                        flag = False
                        break  # Exit the inner while loop
                    random_choice = np.random.randint(0, len(spawn_choices)) 
                    piece_start_pos = spawn_choices[random_choice] 
                
                all_start_positions.append(piece_start_pos)
                if not flag:
                    break
                

            # The search space is not exhausted and all pieces have been spawned successfully
            if flag:
                break
            # else try again

        assert len(all_start_positions) == self.num_pieces, "Number of pieces spawned is not equal to the number of pieces specified"
        return all_start_positions

    def set_board_layout(self, target_shape=None, target_colour=None, level=None):
        # Get all start positions for all pieces on the board
        all_start_positions = self.set_start_positions()
        
        # Set agent start position at the center of the board
        center_sq = math.ceil((self.board_size)/2)
        agent_start_pos = np.array([center_sq, center_sq], dtype=np.int64) # Get start position of agent
        
        grid_info = []
        available_shapes = list(self.shapes)  # List of available shapes
        available_colours = list(self.colours)  # List of available colours

        for i in range(len(all_start_positions)):
            piece_position = all_start_positions[i]

            # Select a random shape from the available shapes
            piece_shape = np.random.choice(available_shapes)
            # Select a random colour from the available colours
            colour_name = np.random.choice(available_colours)

            # Get target_pos
            if i == 0:
                target_pos = piece_position
                if target_shape:
                    piece_shape = target_shape  # Overwrite target shape if specified
                if target_colour:
                    colour_name = target_colour  # Overwrite target colour if specified

            if level == "easy" or level == "sample":
                available_shapes.remove(piece_shape)  # Remove the selected shape from the available shapes
                available_colours.remove(colour_name)  # Remove the selected colour from the available colours
                piece_rotation = 0  # No rotation
            elif level == "medium":
                # Introduce rotation for medium level
                available_shapes.remove(piece_shape)  # Remove the selected shape from the available shapes
                available_colours.remove(colour_name)  # Remove the selected colour from the available colours
                piece_rotation = np.random.randint(0, 4) # Random rotation            
            else:
                # Hard level, allow same shape or colour repitition, based on randomness
                random_value = np.random.randint(0, 2)
                if random_value:
                    available_colours.remove(colour_name)  # Remove the selected colour from the available colours
                else:
                    available_shapes.remove(piece_shape)  # Remove the selected shape from the available shapes

                piece_rotation = np.random.randint(0, 4) # Random rotation

            piece = PentominoPiece(piece_shape, piece_rotation, piece_position)
            piece_grids = piece.get_grid_locations()
            piece_region = layout_utils.get_region(piece_position, self.mapped_regions)
            piece_data = {
                "piece_grids": piece_grids,
                "piece_colour": colour_name,
                "colour_value": COLOURS[colour_name],
                "start_position": piece_position,
                "piece_shape": piece_shape,
                "piece_rotation": piece_rotation,
                "piece_region": piece_region
            }

            grid_info.append(piece_data)

        return agent_start_pos, target_pos, grid_info


if __name__ == '__main__':
    board1 = BoardLayout(board_size=18, num_pieces=4, shapes=PIECE_NAMES, colours=COLOUR_NAMES, seed=640)
    agent_start_pos, target_pos, info = board1.set_board_layout(
        target_shape = 'P',
        target_colour = 'red',
        level = 'easy')
             