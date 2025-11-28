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
from pathlib import Path
from util import nearest_point

#################
# Team creation #
#################

WEIGHTS_FILE_PATH = Path(__file__).parent / 'weights.txt'


def create_team(first_index, second_index, is_red,
                first='OffensiveQAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Returns a list of two agents that will form the team.
    """
    return [eval(first)(first_index, num_training=num_training),
            eval(second)(second_index)]

##########
# Agents #
##########


class QLearningAgent(CaptureAgent):
    """
    A General Q-Learning Agent with NaN protection.
    """

    def __init__(self, index, time_for_computing=.1, num_training=0, epsilon=0.5, alpha=0.01, gamma=0.8):
        CaptureAgent.__init__(self, index, time_for_computing)
        self.episodes_so_far = 0
        self.num_training = int(num_training)
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.discount = gamma   # Discount factor
        self.weights = util.Counter()

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)

        # Helper: Map dimensions for normalization (optional but good for debugging)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height

        # Load weights
        if self.episodes_so_far == 0:
            try:
                with open(WEIGHTS_FILE_PATH, 'r') as file:
                    content = file.read()
                    if content:
                        self.weights = util.Counter(eval(content))
            except IOError:
                # If file doesn't exist, start with empty weights
                self.weights = util.Counter()

    def get_q_value(self, game_state, action):
        """
        Computes Q(state, action) = weights * features
        """
        features = self.get_features(game_state, action)
        return features * self.weights

    def get_value(self, game_state):
        """
        Returns max_action Q(state, action)
        """
        possible_actions = game_state.get_legal_actions(self.index)
        if not possible_actions:
            return 0.0
        return max([self.get_q_value(game_state, a) for a in possible_actions])

    def get_policy(self, game_state):
        """
        Compute the best action to take in a state.
        """
        possible_actions = game_state.get_legal_actions(self.index)
        if not possible_actions:
            return None

        q_values = [self.get_q_value(game_state, a) for a in possible_actions]
        best_value = max(q_values)

        # Randomly choose among the best actions (handling float tolerance)
        best_actions = [a for a, v in zip(
            possible_actions, q_values) if v >= best_value - 1e-9]

        if not best_actions:
            return random.choice(possible_actions)
        return random.choice(best_actions)

    def choose_action(self, game_state):
        """
        Picks an action. 
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        # Training vs Testing
        if self.episodes_so_far < self.num_training:
            if random.random() < self.epsilon:
                action = random.choice(legal_actions)
            else:
                action = self.get_policy(game_state)
        else:
            action = self.get_policy(game_state)

        # Learning update
        if self.episodes_so_far < self.num_training:
            self.update_weights(game_state, action)

        return action

    def update_weights(self, game_state, action):
        """
        The core Q-Learning update with NaN protection.
        """
        next_state = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, next_state)
        features = self.get_features(game_state, action)

        future_q = self.get_value(next_state)
        current_q = self.get_q_value(game_state, action)

        difference = (reward + self.discount * future_q) - current_q

        # --- CRITICAL FIX FOR NaN ---
        # If difference explodes, cap it. This prevents the weights from hitting infinity.
        if abs(difference) > 1000:
            difference = 1000 if difference > 0 else -1000
        # ----------------------------

        for feature in features:
            self.weights[feature] += self.alpha * \
                difference * features[feature]

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def final(self, game_state):
        """
        Called at the end of each game.
        """
        CaptureAgent.final(self, game_state)
        self.episodes_so_far += 1

        print(
            f"Final weights after episode {self.episodes_so_far}: {self.weights}")

        # Check if we are in training mode
        if self.num_training > 0:
            # Print progress so you know it's working
            print(
                f"Training Episode {self.episodes_so_far}/{self.num_training}. Saving weights...")

            # Save EVERY episode, so you don't lose data if you crash/stop
            with open(WEIGHTS_FILE_PATH, 'w', encoding='utf-8') as file:
                file.write(str(self.weights))

    def get_features(self, game_state, action):
        return util.Counter()

    def get_reward(self, game_state, next_state):
        return 0


class OffensiveQAgent(QLearningAgent):
    """
    Tournament-Grade Offensive Agent.
    Features:
    1. Dynamic Mode Switching (Attack vs Retreat).
    2. Capsule Awareness (Hunt capsules to scare ghosts).
    3. Interceptor Logic (Kill invaders if nearby).
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # --- Pre-calculations ---
        food_list = self.get_food(successor).as_list()
        walls = game_state.get_walls()
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]

        # Defenders: Enemies on their side (Ghost mode)
        defenders = [
            a for a in enemies if not a.is_pacman and a.get_position() is not None]
        # Invaders: Enemies on our side (Pacman mode)
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]

        # 1. BIAS
        features['bias'] = 1.0

        # 2. STATE MACHINE: "Should I go home?"
        # Logic: If we have > 2 food, or time is short, or ghost is too close... run home.

        num_carrying = my_state.num_carrying
        timeleft = successor.data.timeleft / 4

        # Distance to nearest dangerous ghost
        dist_to_ghost = 9999
        if len(defenders) > 0:
            dist_to_ghost = min([self.get_maze_distance(
                my_pos, a.get_position()) for a in defenders])

        should_return = False
        if num_carrying > 0:
            if timeleft < 60:
                should_return = True     # Time pressure
            if num_carrying >= 5:
                should_return = True  # Bag full
            if dist_to_ghost < 6:
                should_return = True  # Danger nearby

        # 3. FEATURES BASED ON MODE

        if should_return:
            # === RETREAT MODE ===
            dist_to_home = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = -1 * \
                dist_to_home  # Minimize dist to home

            # Panic if ghost is close while retreating
            if dist_to_ghost <= 5:
                # Maximize distance to ghost (positive weight usually implies good,
                # but Q-learning will learn the sign. We provide raw distance)
                features['distance_to_ghost'] = dist_to_ghost
                if len(game_state.get_legal_actions(self.index)) < 2:
                    features['trapped'] = 1  # Avoid dead ends

        else:
            # === ATTACK MODE ===

            # A. Eat Food
            features['successor_score'] = - \
                len(food_list)  # Maximize food eaten

            # B. Go to nearest Food
            if len(food_list) > 0:
                min_dist = min([self.get_maze_distance(my_pos, food)
                               for food in food_list])
                features['distance_to_food'] = -1 * \
                    min_dist  # Minimize dist to food

            # C. Hunt Capsules (Power pills)
            capsules = self.get_capsules(successor)
            if len(capsules) > 0:
                min_cap_dist = min(
                    [self.get_maze_distance(my_pos, cap) for cap in capsules])
                features['distance_to_capsule'] = -1 * \
                    min_cap_dist * 2  # High priority

            # D. Ghost Handling
            if len(defenders) > 0:
                scared_timers = [a.scared_timer for a in defenders]
                is_scared = any(t > 0 for t in scared_timers)

                if not is_scared:
                    # Avoid active ghosts
                    if dist_to_ghost < 4:
                        features['danger_ghost'] = 1  # Very Bad
                    elif dist_to_ghost < 7:
                        # Maintain distance
                        features['distance_to_ghost'] = dist_to_ghost
                else:
                    # Hunt scared ghosts (treat distance as negative to minimize it)
                    features['distance_to_ghost'] = -1 * dist_to_ghost

        # 4. INTERCEPTOR LOGIC (Opportunity check)
        # If we see an invader nearby, distract ourselves to kill it
        if len(invaders) > 0:
            invader_dist = min([self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders])
            if invader_dist < 6:
                # Go kill them
                features['distance_to_invader'] = -1 * invader_dist

                # Bonus if we are not a pacman (we are on our side)
                if not my_state.is_pacman and my_state.scared_timer == 0:
                    features['kill_invader'] = 1

        # 5. GENERAL MOVEMENT
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_reward(self, game_state, next_state):
        my_state = next_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        reward = 0

        # Reward 1: Eat food
        if my_state.num_carrying > game_state.get_agent_state(self.index).num_carrying:
            reward += 10

        # Reward 2: Return food (Big Reward)
        returned = my_state.num_returned - \
            game_state.get_agent_state(self.index).num_returned
        if returned > 0:
            reward += (returned * 20)

        # Reward 3: Survival
        if my_pos == self.start:
            reward -= 50  # Big penalty for dying

        # Reward 4: Capsules
        if len(self.get_capsules(next_state)) < len(self.get_capsules(game_state)):
            reward += 50

        return reward


class DefensiveReflexAgent(CaptureAgent):
    """
    A simple, reliable defensive agent. 
    It patrols the border and chases invaders.
    """
# TODO If defensive agent is white screened,
# it should try leave the attack zone

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. Defense Status
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # 2. Invader tracking
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # 3. Stop/Reverse penalties
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor
