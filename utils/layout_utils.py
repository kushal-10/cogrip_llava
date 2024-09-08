# Helper functions for layout.py
import math

def valid(N: int, selected_position: list[int], spawned_positions: list[list[int]]) -> bool:
    '''
    Checks if the new selected position of a piece is overlapping with any of the spawned pieces
    Args:
        N - Size of the board
        selected_position - The center grid point of a piece
        spawned_position - A list of center grid points of previously spawned pieces
    '''
    piece_size = 3 # Each piece is a 3x3 grid

    if len(spawned_positions) == 0:
        if selected_position[0] <= N - piece_size + 1 and selected_position[1] <= N - piece_size + 1 and selected_position[0] >= piece_size/2 and selected_position[1] >= piece_size/2:
            return True
        else:
            return False
    
    else:
        for pos in spawned_positions:
            # For each previously assigned piece get the difference between center grid points of each. Atleast 2 grid spaces between two pieces
            diff_x = abs(pos[0] - selected_position[0])
            diff_y = abs(pos[1] - selected_position[1])

            if diff_x >= piece_size and diff_y >= piece_size: # Do not overlap pieces
                # Piece inside boundaries
                if selected_position[0] > N - piece_size + 1 or selected_position[1] > N - piece_size + 1 or selected_position[0] < piece_size/2 or selected_position[1] < piece_size/2: 
                    return False
            else:
                return False

    return True
    
    
def map_regions(N: int) -> dict:
    '''
    Based on the Grid Size of the board, map grid location to its region (top, bottom, top-left ...)
    Args:
        N - Size of the board
    Returns:
        region_map - A dictionary mapping each grid point to it region
    '''

    region_map = {
        'top left': [], 'top': [], 'top right': [], 'left': [], 'center': [], 'right': [], 'bottom left': [], 'bottom': [], 'bottom right': []
    }

    C = math.floor(N/2) # Center grid value
    pad = math.floor(N/6) # 1/6th of the board size - central grid - 1/6 (around center)
    bound1 = C - pad
    bound2 = C + pad

    # Gym has Y axis reveres, X-axis as it is
    for x in range(N):
        for y in range(N):
            if x < bound1 and y < bound1:
                region_map['top left'].append([x, y])
            elif x < bound1 and y >= bound1 and y <= bound2:
                region_map['left'].append([x, y])
            elif x < bound1 and y > bound2:
                region_map['bottom left'].append([x, y])
            elif x >= bound1 and x <= bound2 and y < bound1:
                region_map['top'].append([x, y])
            elif x >= bound1 and x <= bound2 and y > bound2:
                region_map['bottom'].append([x, y])
            elif x > bound2 and y < bound1:
                region_map['top right'].append([x, y])
            elif x > bound2 and y > bound2:
                region_map['bottom right'].append([x, y])
            elif x > bound2 and y >= bound1 and y <= bound2:
                region_map['right'].append([x, y])
            else:
                region_map['center'].append([x, y])
            
    return region_map


def get_region(pos: list[int], region_map: dict) -> str:
    '''
    Get the region of a grid point
    Args:
        pos - The grid point
        region_map - The region map of the board
    Returns:
        The region of the grid point
    '''

    for region, positions in region_map.items():
        if pos in positions:
            return region
    
    print("Position not found in any region")
        
if __name__ == '__main__':
    print(map_regions(9))
    print(map_regions(18))
    print(map_regions(27))