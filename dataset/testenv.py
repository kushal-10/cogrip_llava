from grip_env.environment import GridWorldEnv

board_size = 15
info =  [
            {
                "piece_grids": [
                    [
                        3,
                        7
                    ],
                    [
                        4,
                        7
                    ],
                    [
                        3,
                        8
                    ],
                    [
                        4,
                        8
                    ],
                    [
                        3,
                        9
                    ]
                ],
                "piece_colour": "red",
                "colour_value": [
                    255,
                    0,
                    0
                ],
                "start_position": [
                    3,
                    8
                ],
                "piece_shape": "P",
                "piece_rotation": 0,
                "piece_region": "left"
            },
            {
                "piece_grids": [
                    [
                        12,
                        3
                    ],
                    [
                        13,
                        3
                    ],
                    [
                        13,
                        4
                    ],
                    [
                        13,
                        5
                    ],
                    [
                        14,
                        5
                    ]
                ],
                "piece_colour": "magenta",
                "colour_value": [
                    255,
                    0,
                    255
                ],
                "start_position": [
                    13,
                    4
                ],
                "piece_shape": "Z",
                "piece_rotation": 0,
                "piece_region": "top right"
            },
            {
                "piece_grids": [
                    [
                        5,
                        12
                    ],
                    [
                        6,
                        12
                    ],
                    [
                        7,
                        12
                    ],
                    [
                        6,
                        13
                    ],
                    [
                        6,
                        14
                    ]
                ],
                "piece_colour": "green",
                "colour_value": [
                    0,
                    255,
                    0
                ],
                "start_position": [
                    6,
                    13
                ],
                "piece_shape": "T",
                "piece_rotation": 0,
                "piece_region": "bottom"
            },
            {
                "piece_grids": [
                    [
                        3,
                        10
                    ],
                    [
                        3,
                        11
                    ],
                    [
                        4,
                        11
                    ],
                    [
                        4,
                        12
                    ],
                    [
                        5,
                        12
                    ]
                ],
                "piece_colour": "yellow",
                "colour_value": [
                    255,
                    255,
                    0
                ],
                "start_position": [
                    4,
                    11
                ],
                "piece_shape": "W",
                "piece_rotation": 0,
                "piece_region": "bottom left"
            }
        ]

import numpy as np
agent_start_pos = np.array([8,8])
target_pos = np.array([12,3])

env = GridWorldEnv(render_mode="rgb_array", size=board_size, grid_info=info, agent_pos=agent_start_pos, target_pos=target_pos)
env.reset()
image = env.render()