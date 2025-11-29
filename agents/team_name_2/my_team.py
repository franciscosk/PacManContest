# my_team.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
from pathlib import Path

#################
# Team creation #
#################

# File path for weights
WEIGHTS_FILE_PATH = Path(__file__).parent / 'weights.txt'


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexQAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Entry point called by the capture framework.
    """
    return [eval(first)(first_index, num_training=num_training),
            # rEFLEX AGENT doesnt have num_training [param] yet
            eval(second)(second_index)]


##########
# Agents #
##########


class ApproximateQAgent(CaptureAgent):
    """
    Generic approximate Q-learning agent.
    """

    def __init__(self, index, time_for_computing=.1, num_training=0,
                 epsilon=0.1, alpha=0.2, gamma=0.8):
        CaptureAgent.__init__(self, index, time_for_computing)
        self.episodes_so_far = 0
        self.num_training = int(num_training)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

        # Linear function approximation weights
        self.weights = util.Counter()

        # Small bit of memory to detect oscillations / revisits
        self.last_positions = []
        self.max_position_history = 6

    def assert_finite_counter(self, counter, label):
        import math
        for k, v in counter.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                raise ValueError(f"{label} has bad value: {k} = {v}")

    def register_initial_state(self, game_state):

        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.last_positions = [self.start]

        # --- WEIGHT LOADING LOGIC ---
        loaded = False
        try:
            if WEIGHTS_FILE_PATH.exists():
                with open(WEIGHTS_FILE_PATH, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    self.weights = util.Counter(eval(content))
                    loaded = True
        except OSError:
            pass

        if not loaded or len(self.weights) == 0:
            self.weights = self.get_initial_weights()
            print(f"[ApproxQ] Using initial weights: {self.weights}")
        else:
            print(f"[ApproxQ] Loaded weights: {self.weights}")

        # *** NEW: sanity check ***
        self.assert_finite_counter(self.weights, "weights after load")

    def final(self, game_state):
        CaptureAgent.final(self, game_state)
        self.episodes_so_far += 1

        # --- SAVING LOGIC ENABLED FOR TRAINING ---
        print(f"New weights: {self.weights}")
        if self.num_training > 0:
            print("Saving weights to file...")
            # Only save occasionally to avoid file corruption if interrupted,
            # or save at the very end.
            with open(WEIGHTS_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(str(self.weights))

    def get_initial_weights(self):
        return util.Counter()

    def get_features(self, game_state, action):
        return util.Counter()

    def get_reward(self, game_state, next_state):
        return 0.0

    def get_q_value(self, game_state, action):
        features = self.get_features(game_state, action)
        return self.weights * features

    def get_value(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return 0.0
        return max(self.get_q_value(game_state, a) for a in actions)

    def get_policy(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None
        values = [self.get_q_value(game_state, a) for a in actions]
        best_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == best_value]

        if len(best_actions) == 0:
            return random.choice(actions)

        return random.choice(best_actions)

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP

        exploring = self.episodes_so_far < self.num_training
        if exploring and random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            action = self.get_policy(game_state)

        if exploring:
            self.update_weights(game_state, action)

        successor = self.get_successor(game_state, action)
        next_pos = successor.get_agent_state(self.index).get_position()
        if next_pos is not None:
            self.last_positions.append(next_pos)
            if len(self.last_positions) > self.max_position_history:
                self.last_positions.pop(0)

        return action

    def update_weights(self, game_state, action):
        next_state = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, next_state)
        features = self.get_features(game_state, action)
        current_q = self.get_q_value(game_state, action)
        future_q = self.get_value(next_state)

        difference = (reward + self.discount * future_q) - current_q

        # --- FROZEN FEATURES LOGIC ---
        # These weights will NEVER change during training.
        frozen_features = ['ghost_1_step',
                           'ghost_2_steps', 'trapped_location', 'stop']

        for f in features:
            if f not in frozen_features:
                self.weights[f] += self.alpha * difference * features[f]

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            successor = successor.generate_successor(self.index, action)
        return successor


# ----------------------------------------------------------------------
# OFFENSIVE AGENT
# ----------------------------------------------------------------------

class OffensiveReflexQAgent(ApproximateQAgent):
    """
    Improved Offensive Agent with "Shaping" for training.
    Safety weights are frozen. Strategy weights are learnable.
    """

    def get_initial_weights(self):
        w = util.Counter()
        w['bias'] = 0.0

        # --- LEARNABLE WEIGHTS (Strategy) ---
        # We start these at reasonable guesses, but let the agent tune them.
        w['closest_food'] = -2.0
        w['eat_food'] = 20.0
        # Start neutral, let it learn if carrying is good/bad
        w['carrying'] = 0.0
        w['go_home'] = -5.0             # Distance penalty to home
        w['eat_capsule'] = 50.0
        w['ghost_far'] = 0.0
        w['reverse'] = -2.0
        w['revisit'] = -5.0

        # --- FROZEN WEIGHTS (Safety - DO NOT CHANGE) ---
        # These are critical for survival. We do not want the agent to "experiment" with these.
        w['ghost_1_step'] = -1000.0
        w['ghost_2_steps'] = -1000.0
        w['trapped_location'] = -5000.0
        w['stop'] = -100.0

        return w

    def get_features(self, game_state, action):
        features = util.Counter()

        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 0. Bias
        features['bias'] = 1.0

        # 1. Food
        food_list = self.get_food(successor).as_list()
        features['closest_food'] = 0.0
        if len(food_list) > 0:
            min_dist = min(self.get_maze_distance(my_pos, f)
                           for f in food_list)
            features['closest_food'] = float(min_dist)

        # 2. Eat Food (Reward for decreasing food count)
        current_food = self.get_food(game_state).as_list()
        if len(food_list) < len(current_food):
            features['eat_food'] = 1.0

        # 3. Carrying logic & Retreat
        # We define a "retreat" condition to encourage going home.
        should_retreat = False
        carrying_amt = my_state.num_carrying
        timeleft = game_state.data.timeleft

        # Retreat if we have 2+ dots, or 1 dot and time is short
        if carrying_amt >= 2:
            should_retreat = True
        elif carrying_amt > 0 and timeleft < 100:
            should_retreat = True

        if should_retreat:
            dist_home = self.get_maze_distance(my_pos, self.start)
            features['go_home'] = float(dist_home)
            # If retreating, minimize the urge to go deeper for food
            features['closest_food'] = 0.0

        # 4. Ghost handling & Trapped Logic
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if (
            not a.is_pacman) and a.get_position() is not None]

        features['ghost_1_step'] = 0.0
        features['ghost_2_steps'] = 0.0
        features['trapped_location'] = 0.0

        if defenders:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in defenders]
            min_ghost_dist = min(dists)
            scared = any(a.scared_timer > 0 for a in defenders)

            if not scared:
                # Standard avoidance
                if min_ghost_dist <= 1:
                    features['ghost_1_step'] = 1.0
                elif min_ghost_dist <= 3:
                    features['ghost_2_steps'] = 1.0

                # Dead-end detection ("Trapped")
                if min_ghost_dist <= 5:
                    # Check legal moves from the successor state
                    next_actions = successor.get_legal_actions(self.index)
                    move_options = [
                        a for a in next_actions if a != Directions.STOP]

                    # If we have 1 or fewer exits, it's a dead end
                    if len(move_options) <= 1:
                        features['trapped_location'] = 1.0

        # 5. Capsules
        capsules_before = self.get_capsules(game_state)
        capsules_after = self.get_capsules(successor)
        if len(capsules_after) < len(capsules_before):
            features['eat_capsule'] = 1.0

        # 6. Movement Penalties
        if action == Directions.STOP:
            features['stop'] = 1.0

        current_direction = game_state.get_agent_state(
            self.index).configuration.direction
        if current_direction != Directions.STOP:
            reverse_dir = Directions.REVERSE[current_direction]
            if action == reverse_dir:
                features['reverse'] = 1.0

        if my_pos in self.last_positions:
            features['revisit'] = 1.0

        return features

    def get_reward(self, game_state, next_state):
        reward = 0.0
        my_prev = game_state.get_agent_state(self.index)
        my_next = next_state.get_agent_state(self.index)

        # Returning food is the ultimate goal
        if my_next.num_returned > my_prev.num_returned:
            reward += 100.0 * (my_next.num_returned - my_prev.num_returned)

        # Dying is bad
        if my_prev.num_carrying > 0 and my_next.num_carrying == 0:
            if next_state.get_agent_position(self.index) == self.start:
                reward -= 100.0

        return reward


# ----------------------------------------------------------------------
# DEFENSIVE AGENT
# ----------------------------------------------------------------------
class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Hardcoded Defense Logic: if food is scarce, guard the border tightly
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

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
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders]
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
