from pacman import Directions
from game import Agent
import api
import random
import game
import util


def get_parameters(width, height):
    grid_size = width * height
    if grid_size <= 100:
        # Small layout parameters
        DANGER_PENALTY = -50
        GAMMA = 0.93
        DELTA = 0.001

        # Rewards for small layouts
        REWARD_FOOD = 5
        REWARD_CAPSULE = 0
        REWARD_GHOST = -100
        REWARD_EAT_GHOST = 0
        REWARD_EMPTY_CELL = 0
        SCARED_GHOST_REWARD_FACTOR = 0

        # Fixed safety reward for open spaces
        REWARD_OPEN = 1

        # Parameters for dynamic danger radius in small grids
        DANGER_RADIUS_CLOSE = 1
        DANGER_RADIUS_MEDIUM = 0
        DANGER_RADIUS_FAR = 0
        DISTANCE_THRESHOLD_CLOSE = 4
        DISTANCE_THRESHOLD_MEDIUM = 0
    else:
        # Medium/Large layout parameters
        DANGER_PENALTY = -500
        GAMMA = 0.93
        DELTA = 0.001

        # Rewards for medium/large layouts
        REWARD_FOOD = 10
        REWARD_CAPSULE = 20
        REWARD_GHOST = -1000
        REWARD_EAT_GHOST = 0
        REWARD_EMPTY_CELL = 0
        SCARED_GHOST_REWARD_FACTOR = 20

        # Fixed safety reward for open spaces
        REWARD_OPEN = 1

        # Parameters for dynamic danger radius in medium/large grids
        DANGER_RADIUS_CLOSE = 3
        DANGER_RADIUS_MEDIUM = 2
        DANGER_RADIUS_FAR = 1
        DISTANCE_THRESHOLD_CLOSE = 5
        DISTANCE_THRESHOLD_MEDIUM = 10

    return (DANGER_PENALTY, GAMMA, DELTA,
            REWARD_FOOD, REWARD_CAPSULE, REWARD_GHOST, REWARD_EAT_GHOST,
            REWARD_EMPTY_CELL, SCARED_GHOST_REWARD_FACTOR,
            DANGER_RADIUS_CLOSE, DANGER_RADIUS_MEDIUM, DANGER_RADIUS_FAR,
            DISTANCE_THRESHOLD_CLOSE, DISTANCE_THRESHOLD_MEDIUM, REWARD_OPEN)


class MDPAgent(Agent):
    def __init__(self):
        self.corners = None
        self.width = None
        self.height = None
        self.map = None
        self.DANGER_PENALTY = None
        self.GAMMA = None
        self.DELTA = None
        self.walls = set()

        self.REWARD_FOOD = None
        self.REWARD_CAPSULE = None
        self.REWARD_GHOST = None
        self.REWARD_EAT_GHOST = None
        self.REWARD_EMPTY_CELL = None
        self.SCARED_GHOST_REWARD_FACTOR = None

        self.DANGER_RADIUS_CLOSE = None
        self.DANGER_RADIUS_MEDIUM = None
        self.DANGER_RADIUS_FAR = None
        self.DISTANCE_THRESHOLD_CLOSE = None
        self.DISTANCE_THRESHOLD_MEDIUM = None

        self.REWARD_OPEN = None

    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:", api.whereAmI(state))

        walls_list = api.walls(state)
        self.walls = set(walls_list)
        self.corners = api.corners(state)
        self.width = max(x for x, y in self.walls) + 1
        self.height = max(y for x, y in self.walls) + 1

        parameters = get_parameters(self.width, self.height)
        (self.DANGER_PENALTY, self.GAMMA, self.DELTA,
         self.REWARD_FOOD, self.REWARD_CAPSULE, self.REWARD_GHOST,
         self.REWARD_EAT_GHOST, self.REWARD_EMPTY_CELL,
         self.SCARED_GHOST_REWARD_FACTOR,
         self.DANGER_RADIUS_CLOSE, self.DANGER_RADIUS_MEDIUM, self.DANGER_RADIUS_FAR,
         self.DISTANCE_THRESHOLD_CLOSE, self.DISTANCE_THRESHOLD_MEDIUM,
         self.REWARD_OPEN) = parameters

        self.map = initial_map(self.width, self.height, self.walls)

    def final(self, state):
        print("Looks like the game just ended!")

    def getAction(self, state):
        self.map = value_iteration(self, self.map, state)
        legal = api.legalActions(state)

        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman_position = api.whereAmI(state)
        x, y = pacman_position
        scores, actions = evaluate_actions(legal, self.map, x, y, self.height, self.width)
        if not scores:
            choice = random.choice(legal)
        else:
            max_score_index = scores.index(max(scores))
            choice = actions[max_score_index]

        return api.makeMove(choice, legal)


def evaluate_actions(legal, utility_map, x, y, height, width):
    scores = []
    actions = []
    for action in legal:
        if action == Directions.NORTH and y + 1 < height:
            value = utility_map[x][y + 1]
        elif action == Directions.SOUTH and y - 1 >= 0:
            value = utility_map[x][y - 1]
        elif action == Directions.EAST and x + 1 < width:
            value = utility_map[x + 1][y]
        elif action == Directions.WEST and x - 1 >= 0:
            value = utility_map[x - 1][y]
        else:
            value = None
        if value is not None:
            scores.append(value)
            actions.append(action)

    return scores, actions


def value_iteration(agent, utility_map, state):
    food = api.food(state)
    walls_set = agent.walls
    ghost_states = api.ghostStatesWithTimes(state)
    capsules = api.capsules(state)
    width = agent.width
    height = agent.height

    reward_map = create_reward_map(width, height, walls_set, food, capsules, agent)

    apply_ghost_penalties(reward_map, ghost_states, walls_set, width, height,
                          agent.DANGER_PENALTY, agent, state)

    delta = float('inf')  # Initialize delta to infinity
    iteration_count = 0
    while delta >= agent.DELTA:
        delta = 0
        new_utility_map = [[0 for y in range(height)] for x in range(width)]
        for x in range(width):
            for y in range(height):
                if (x, y) in walls_set:
                    new_utility_map[x][y] = None
                else:
                    r = reward_map[x][y]
                    u = bellman_equation(utility_map, (x, y), width, height, r, agent.GAMMA)
                    new_utility_map[x][y] = u
                    delta = max(delta, abs(u - utility_map[x][y]))
        utility_map = new_utility_map
        iteration_count += 1
    return utility_map


def bellman_equation(utility_map, cell, width, height, r, gamma):
    x, y = cell

    north = utility_map[x][y + 1] if y + 1 < height and utility_map[x][y + 1] is not None else utility_map[x][y]
    south = utility_map[x][y - 1] if y - 1 >= 0 and utility_map[x][y - 1] is not None else utility_map[x][y]
    east = utility_map[x + 1][y] if x + 1 < width and utility_map[x + 1][y] is not None else utility_map[x][y]
    west = utility_map[x - 1][y] if x - 1 >= 0 and utility_map[x - 1][y] is not None else utility_map[x][y]

    utility_north = 0.8 * north + 0.1 * east + 0.1 * west
    utility_south = 0.8 * south + 0.1 * east + 0.1 * west
    utility_east = 0.8 * east + 0.1 * north + 0.1 * south
    utility_west = 0.8 * west + 0.1 * north + 0.1 * south

    max_utility = max(utility_north, utility_south, utility_east, utility_west)
    return r + gamma * max_utility


def apply_ghost_penalties(reward_map, ghost_states, walls_set, width, height,
                          danger_penalty, agent, state):
    pacman_x, pacman_y = api.whereAmI(state)

    for ghost_info in ghost_states:
        (gx, gy), scaredTimer = ghost_info
        gx, gy = int(gx), int(gy)
        if (gx, gy) not in walls_set and reward_map[gx][gy] is not None:
            distance_to_pacman = abs(gx - pacman_x) + abs(gy - pacman_y)
            dynamic_danger_radius = calculate_dynamic_radius(
                distance_to_pacman,
                agent.DANGER_RADIUS_CLOSE,
                agent.DANGER_RADIUS_MEDIUM,
                agent.DANGER_RADIUS_FAR,
                agent.DISTANCE_THRESHOLD_CLOSE,
                agent.DISTANCE_THRESHOLD_MEDIUM
            )

            if scaredTimer > 1 and agent.SCARED_GHOST_REWARD_FACTOR >= 0:
                scared_ghost_reward = (scaredTimer * agent.SCARED_GHOST_REWARD_FACTOR
                                       + agent.REWARD_EAT_GHOST)
                reward_map[gx][gy] = scared_ghost_reward
            else:
                reward_map[gx][gy] = agent.REWARD_GHOST
                nearby_cells = get_cells_within_radius((gx, gy), dynamic_danger_radius, width, height, walls_set)
                for cell in nearby_cells:
                    x, y = cell
                    if reward_map[x][y] is not None:
                        reward_map[x][y] += danger_penalty  # Apply constant negative penalty


def calculate_dynamic_radius(distance_to_pacman,
                             radius_close, radius_medium, radius_far,
                             threshold_close, threshold_medium):
    if distance_to_pacman <= threshold_close:
        return radius_close
    elif distance_to_pacman <= threshold_medium:
        return radius_medium
    else:
        return radius_far


def get_cells_within_radius(center, radius, width, height, walls_set):
    x_center, y_center = center
    cells = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = x_center + dx, y_center + dy
            if 0 <= x < width and 0 <= y < height and (x, y) not in walls_set:
                cells.append((x, y))
    return cells


def create_reward_map(width, height, walls_set, food_positions, capsule_positions, agent):
    reward_map = [[agent.REWARD_EMPTY_CELL for _ in range(height)] for _ in range(width)]

    for x, y in walls_set:
        reward_map[x][y] = None

    for x, y in food_positions:
        reward_map[x][y] = agent.REWARD_FOOD
    for x, y in capsule_positions:
        reward_map[x][y] = agent.REWARD_CAPSULE

    for x in range(width):
        for y in range(height):
            if reward_map[x][y] is not None and (x, y) not in capsule_positions:
                potential_paths = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                valid_paths = [
                    p for p in potential_paths
                    if 0 <= p[0] < width and 0 <= p[1] < height and p not in walls_set
                ]

                if len(valid_paths) >= 3:
                    reward_map[x][y] += agent.REWARD_OPEN

    return reward_map


def initial_map(width, height, walls_set):
    utility_map = [[0 for _ in range(height)] for _ in range(width)]
    for x, y in walls_set:
        utility_map[x][y] = None
    return utility_map
