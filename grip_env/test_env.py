from grip_env.environment import GridWorldEnv
from grip_env.layout import BoardLayout
from grip_env.pieces import PIECE_NAMES, COLOUR_NAMES

if __name__ == '__main__':
    board1 = BoardLayout(board_size=18, num_pieces=4, shapes=PIECE_NAMES, colours=COLOUR_NAMES, seed=640)
    agent_start_pos, target_pos, info = board1.set_board_layout(
        target_shape = 'P',
        target_colour = 'red',
        level = 'easy') 
    
    env = GridWorldEnv(render_mode="human", size=18, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
    env.reset()
    env.render()
    for i in range(1000):
        # RIGHT, DOWN, LEFT, UP
        env.step(0)
        env.render()
        env.step(0)
        env.render()
        env.step(0)
        env.render()
        env.step(1)
        env.render()
        env.step(1)
        env.render()
        env.step(2)
        env.render()
        env.step(2)
        env.render()
        env.step(3)
        env.render()
        env.step(3)
        env.render()
