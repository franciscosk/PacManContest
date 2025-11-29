# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
from pathlib import Path
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='HybridQLearningAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Returns the agents.
    """
    return [eval(first)(first_index, num_training=num_training), eval(second)(second_index)]


##########
# Agents #
##########

class HybridQLearningAgent(CaptureAgent):
    """
    A Hybrid Agent:
    1. ATTACK (Q-Learning): Uses learned weights to gather food safely.
    2. RETREAT (A* Search): Uses strict pathfinding (Ghosts = Walls) to return home.
    """

    def __init__(self, index, time_for_computing=.1, num_training=0):
        super().__init__(index, time_for_computing)
        self.epsilon = 0.05  # Exploration rate
        self.alpha = 0.2     # Learning rate
        self.discount = 0.8  # Discount factor
        self.num_training = num_training
        self.games_played = 0

        self.win = 0
        # Load the "Brain" for Attack Mode
        self.weights_file = Path(__file__).parent / "weights.txt"
        self.weights = self.load_weights()

        # Features we don't want to change during training
        self.frozen_features = ['stop', 'reverse', 'ghost_proximity']

        self.last_state = None
        self.last_action = None
        self.last_score = 0

    def load_weights(self):
        """ Loads weights or sets defaults for Attack Mode """
        if self.weights_file.exists():
            try:
                with open(self.weights_file, "r") as f:
                    return eval(f.read())
            except:
                pass
        # Default Attack Strategy
        return util.Counter({
            'successor_score': 100,
            'distance_to_food': -2,
            'ghost_proximity': -1000,
            'stop': -100,
            'reverse': -2,
            'bias': 1.0
        })

    def save_weights(self):
        # Only save if we actually trained
        if self.num_training > 0:
            with open(self.weights_file, "w") as f:
                f.write(str(self.weights))

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start_pos = game_state.get_agent_position(self.index)
        self.last_score = self.get_score(game_state)

        # Identify Home Boundary
        layout = game_state.get_walls()
        self.boundary = []
        mid_x = layout.width // 2

        # If red, we want to reach mid_x - 1 (left side).
        # If blue, we want to reach mid_x (right side).
        boundary_x = mid_x - 1 if self.red else mid_x

        for y in range(layout.height):
            if not layout[boundary_x][y]:
                self.boundary.append((boundary_x, y))

    def choose_action(self, game_state):
        # 1. ANALYZE STATE
        my_state = game_state.get_agent_state(self.index)
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())

        # 2. DECIDE MODE
        # Mode: RETREAT (A*)
        # Trigger: We have 3+ food OR only 2 dots left on map
        if carrying >= 3 or food_left <= 2:
            safe_action = self.find_safe_path_home(game_state)
            if safe_action is not None:
                # If A* finds a path, take it.
                self.last_state = game_state
                self.last_action = safe_action
                return safe_action

        # Mode: ATTACK (Q-Learning)

        # --- Q-LEARNING UPDATE ---
        # Only update weights if we are in training mode!
        if self.num_training > 0 and self.last_state is not None:
            reward = self.get_score(game_state) - self.last_score
            self.last_score = self.get_score(game_state)
            self.update(self.last_state, self.last_action, game_state, reward)

        # --- ACTION SELECTION ---
        actions = game_state.get_legal_actions(self.index)

        # Epsilon greedy only during training
        if self.num_training > 0 and util.flip_coin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.get_best_action(game_state, actions)

        self.last_state = game_state
        self.last_action = action
        return action

    def find_safe_path_home(self, game_state):
        """
        Performs A* Search to the nearest boundary.
        CRITICAL: Treats enemies as WALLS.
        """
        # FIX: Use nearest_point to snap to grid
        raw_pos = game_state.get_agent_position(self.index)
        start_pos = nearest_point(raw_pos)

        # 1. Identify "Dangerous" coordinates (Ghosts)
        enemies = [game_state.get_agent_state(
            i) for i in self.get_opponents(game_state)]
        danger_spots = set()

        for enemy in enemies:
            if not enemy.is_pacman and enemy.get_position() and enemy.scared_timer <= 0:
                e_pos = nearest_point(enemy.get_position())
                ex, ey = int(e_pos[0]), int(e_pos[1])

                # Mark the ghost's exact spot as dangerous
                danger_spots.add((ex, ey))

                # Mark neighbors as dangerous too (buffer zone)
                neighbors = [(ex+1, ey), (ex-1, ey), (ex, ey+1), (ex, ey-1)]
                for n in neighbors:
                    danger_spots.add(n)

        # 2. Define A* Components
        # PriorityQueue stores (priority, (position, first_action, cost_g))
        queue = util.PriorityQueue()

        # Push start: (pos, action_taken, cost_so_far)
        queue.push((start_pos, None, 0), 0)

        visited = set()  # (x,y)

        while not queue.is_empty():
            curr_pos, first_action, g = queue.pop()

            if curr_pos in visited:
                continue
            visited.add(curr_pos)

            # GOAL CHECK: Are we at the boundary?
            if curr_pos in self.boundary:
                return first_action

            # EXPAND SUCCESSORS
            x, y = int(curr_pos[0]), int(curr_pos[1])
            candidates = [
                ((x+1, y), Directions.EAST),
                ((x-1, y), Directions.WEST),
                ((x, y+1), Directions.NORTH),
                ((x, y-1), Directions.SOUTH)
            ]

            for next_pos, action in candidates:
                # Check walls
                if not game_state.has_wall(next_pos[0], next_pos[1]):
                    # Check Ghosts
                    if next_pos not in danger_spots and next_pos not in visited:

                        # Cost (g): +1 per step
                        new_g = g + 1

                        # Heuristic (h): Manhattan distance to NEAREST boundary point
                        h = min([util.manhattan_distance(next_pos, b)
                                for b in self.boundary])

                        # Priority f = g + h
                        priority = new_g + h

                        # Track the FIRST action that led to this path
                        new_first_action = first_action if first_action is not None else action

                        queue.push(
                            (next_pos, new_first_action, new_g), priority)

        # If no path found (trapped), return None and let Q-Learning handle it
        return None

    def get_best_action(self, game_state, actions):
        best_val = float('-inf')
        best_actions = []
        for action in actions:
            val = self.evaluate(game_state, action)
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)

        if not best_actions:
            return random.choice(actions)

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        return self.get_features(game_state, action) * self.weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # FIX: Use nearest_point to avoid "Positions not in grid"
        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        features['bias'] = 1.0

        # Feature: Food (Attack Mode)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if len(food_list) > 0:
            # FIX: Calculate distance safely
            dists = []
            for food in food_list:
                try:
                    dists.append(self.get_maze_distance(my_pos, food))
                except:
                    # Fallback to Manhattan if maze distance fails
                    # print("Maze distance failed, using Manhattan distance.")
                    dists.append(util.manhattan_distance(my_pos, food))
            features['distance_to_food'] = min(dists)

        # Feature: Ghost Safety
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position(
        ) and a.scared_timer <= 0]

        if len(defenders) > 0:
            dists = []
            for a in defenders:
                # FIX: Snap defender positions to grid
                apos = nearest_point(a.get_position())
                try:
                    dists.append(self.get_maze_distance(my_pos, apos))
                except:
                    dists.append(util.manhattan_distance(my_pos, apos))

            min_dist = min(dists)

            if min_dist < 2:
                features['ghost_proximity'] = -1000
            elif min_dist < 5:
                features['ghost_proximity'] = min_dist * 10

            # Additional check: don't get trapped
            if len(successor.get_legal_actions(self.index)) <= 1 and min_dist < 5:
                features['ghost_proximity'] = -2000

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def update(self, state, action, next_state, reward):
        # Max Q(s')
        next_actions = next_state.get_legal_actions(self.index)
        max_next_q = max([self.evaluate(next_state, a)
                         for a in next_actions]) if next_actions else 0

        current_q = self.evaluate(state, action)
        difference = (reward + self.discount * max_next_q) - current_q

        features = self.get_features(state, action)
        for feature, value in features.items():
            if feature not in self.frozen_features:
                self.weights[feature] += self.alpha * difference * value

    def final(self, game_state):
        if self.num_training > 0 and self.last_state:
            reward = self.get_score(game_state) - self.last_score
            self.update(self.last_state, self.last_action, game_state, reward)
        self.save_weights()
        self.games_played += 1
        if self.get_score(game_state) > 0:
            self.win += 1
        print(
            f"Agent {self.index}! Total wins: {self.win} out of {self.games_played} games.")
        CaptureAgent.final(self, game_state)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor


class DefensiveReflexAgent(CaptureAgent):
    """
    Standard Defensive Agent
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        if not values:
            return random.choice(actions)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        if not best_actions:
            return random.choice(actions)

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # FIX: Use nearest_point
        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]

        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = []
            for a in invaders:
                # FIX: Snap invader positions to grid and handle errors
                apos = nearest_point(a.get_position())
                try:
                    dists.append(self.get_maze_distance(my_pos, apos))
                except:
                    dists.append(util.manhattan_distance(my_pos, apos))
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
