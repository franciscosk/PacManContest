# my_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
import util
import json
from pathlib import Path
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################


def create_team(first_index, second_index, is_red,
                first='SmartQAgent', second='SmartQAgent', num_training=0):
    """
    Returns a list of two agents that will form the team.
    """
    return [eval(first)(first_index, num_training), eval(second)(second_index, num_training)]

##########
# Agents #
##########


class SmartQAgent(CaptureAgent):
    """
    STABILIZED Q-Learning Agent.
    Prevents weight explosion via:
    1. Feature Normalization (0.0 to 1.0)
    2. Weight Clamping (Hard limits)
    3. Small Rewards
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.epsilon = 0.05
        self.alpha = 0.1     # REDUCED: Lower learning rate for stability
        self.discount = 0.8
        self.episodes_so_far = 0  # <--- Initialize the game counter here

        self.won_games = 0
        self.lost_games = 0
        self.scores = []

        self.weights_file = Path(__file__).parent / "weights.txt"
        self.weights = util.Counter()

        # Heuristics that should not change
        self.frozen_weights = ['stop', 'reverse', 'bias', 'on_defense']

        self.load_weights()

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.last_state = None
        self.last_action = None
        # Cache grid size for normalization
        self.map_area = game_state.data.layout.width * game_state.data.layout.height

    def load_weights(self):
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.weights[k] = float(v)  # Ensure float
            except:
                self.initialize_weights()
        else:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize with sane, small numbers
        self.weights['bias'] = 1.0
        self.weights['score'] = 0.5
        self.weights['food_eaten'] = 2.0
        self.weights['stop'] = -5.0
        self.weights['reverse'] = -0.5
        self.weights['distance_to_food'] = -1.5
        self.weights['distance_to_ghost'] = 2.0
        self.weights['invader_distance'] = -2.0
        self.weights['num_invaders'] = -50.0

    def save_weights(self):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=4)
        except:
            pass

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['bias'] = 1.0

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # --- NORMALIZATION CONSTANTS ---
        # We divide distances by map_area or max_width to keep features < 1.0
        max_dist = self.map_area / 2.0

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        current_score = self.get_score(game_state)
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]
        defenders = [
            a for a in enemies if not a.is_pacman and a.get_position() is not None]

        # 1. CAMPING (Winning > 2)
        if current_score > 2:
            features['on_defense'] = 1
            if my_state.is_pacman:
                features['on_defense'] = 0

            features['num_invaders'] = len(invaders)
            if len(invaders) > 0:
                dists = [self.get_maze_distance(
                    my_pos, a.get_position()) for a in invaders]
                # NORMALIZE: 0.0 to 1.0
                features['invader_distance'] = min(dists) / max_dist

        # 2. OFFENSE / RECOVERY
        else:
            if len(invaders) > 0:
                features['num_invaders'] = len(invaders)
                dists = [self.get_maze_distance(
                    my_pos, a.get_position()) for a in invaders]
                features['invader_distance'] = min(dists) / max_dist
                features['on_defense'] = 1 if not my_state.is_pacman else 0

            else:
                food_list = self.get_food(successor).as_list()
                # DO NOT use raw score here. Use food count fraction.
                features['food_remaining'] = len(food_list) / 100.0

                if len(food_list) > 0:
                    min_dist = min([self.get_maze_distance(my_pos, food)
                                   for food in food_list])
                    features['distance_to_food'] = min_dist / max_dist

                if len(defenders) > 0:
                    dists = [self.get_maze_distance(
                        my_pos, a.get_position()) for a in defenders]
                    min_ghost_dist = min(dists)
                    if min_ghost_dist < 5:
                        if my_state.is_pacman:
                            # Inverse distance proxy (max 1.0)
                            features['distance_to_ghost'] = - \
                                1.0 + (min_ghost_dist / 10.0)

        return features

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def get_q_value(self, game_state, action):
        features = self.get_features(game_state, action)
        return features * self.weights

    def compute_value_from_q_values(self, game_state):
        allowed = game_state.get_legal_actions(self.index)
        if not allowed:
            return 0.0
        return max([self.get_q_value(game_state, a) for a in allowed])

    def compute_action_from_q_values(self, game_state):
        allowed = game_state.get_legal_actions(self.index)
        if not allowed:
            return None
        best_val = -float('inf')
        best_actions = []
        for action in allowed:
            val = self.get_q_value(game_state, action)
            if val > best_val:
                best_val = val
                best_actions = [action]
            elif val == best_val:
                best_actions.append(action)
        return random.choice(best_actions)

    def update(self, game_state, action, next_state, reward):
        features = self.get_features(game_state, action)
        q_value = self.get_q_value(game_state, action)
        next_max_q = self.compute_value_from_q_values(next_state)

        difference = (reward + self.discount * next_max_q) - q_value

        # --- SAFETY CLIP ---
        # Prevent differences from being astronomical
        if difference > 100:
            difference = 100
        if difference < -100:
            difference = -100

        for feature, value in features.items():
            if feature not in self.frozen_weights:
                new_weight = self.weights[feature] + \
                    self.alpha * difference * value

                # --- WEIGHT CLAMPING ---
                # Force weights to stay within [-1000, 1000] to prevent explosion
                if new_weight > 1000:
                    new_weight = 1000
                if new_weight < -1000:
                    new_weight = -1000

                self.weights[feature] = new_weight

    def get_reward(self, game_state, next_state):
        reward = 0
        my_state = next_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        prev_state = game_state.get_agent_state(self.index)

        # SCALED DOWN REWARDS
        score_diff = self.get_score(next_state) - self.get_score(game_state)
        reward += score_diff * 1.0  # Reduced from 10 to 1

        if my_state.num_carrying > prev_state.num_carrying:
            reward += 0.5  # Reduced from 5

        if my_pos == self.start and prev_state.get_position() != self.start:
            reward -= 5.0  # Reduced from 50

        return reward

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        if self.last_state and self.last_action:
            reward = self.get_reward(self.last_state, game_state)
            self.update(self.last_state, self.last_action, game_state, reward)

        if util.flip_coin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.compute_action_from_q_values(game_state)

        self.last_state = game_state
        self.last_action = action
        return action

    def final(self, game_state):
        if self.last_state and self.last_action:
            reward = self.get_reward(self.last_state, game_state)
            self.update(self.last_state, self.last_action, game_state, reward)
        self.save_weights()
        self.episodes_so_far += 1
        self.won_games += 1 if self.get_score(game_state) > 0 else 0
        self.lost_games += 1 if self.get_score(game_state) <= 0 else 0
        self.scores.append(self.get_score(game_state))

        if self.episodes_so_far % 10 == 0:
            print(
                f"Game Over. Weights saved. Score: {self.get_score(game_state)}")

            print(
                f"Total Games so far: {self.episodes_so_far}, Wins: {self.won_games}, Losses: {self.lost_games}")
            print(
                f"Average Score over last {self.episodes_so_far} games: {sum(self.scores)/len(self.scores):.2f}")
        CaptureAgent.final(self, game_state)
