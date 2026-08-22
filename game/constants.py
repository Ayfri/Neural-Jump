# Agent settings
AGENT_VISION_DISTANCE = 3  # Number of tiles the agent can see in each direction

# Movement directions, the first three are the network's outputs
MOVE_JUMP = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
MOVE_IDLE = 3  # Stand still, only ever played by a human: the policy has no output for it
