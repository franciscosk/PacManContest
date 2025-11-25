# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """
    return [eval(first)(first_index, num_training=num_training), eval(second)(second_index, num_training=num_training)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1, num_training=0):
        super().__init__(index, time_for_computing)
        self.start = None
        self.epsilon = 0.05
        self.alpha = 0.001
        self.discount = 0.8
        self.num_training = num_training
        self.episodes_so_far = 0
        self.weights = util.Counter()
        
        # Variables for learning
        self.last_state = None
        self.last_action = None
        self.last_score = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.last_state = None
        self.last_action = None
        self.last_score = self.get_score(game_state)

    def get_q_value(self, game_state, action):
        """
        Computes the Q-value for a given state and action using the weights.
        """
        features = self.get_features(game_state, action)
        return features * self.weights

    def update(self, game_state, action, next_state, reward):
        """
        Updates the weights based on the transition.
        """
        features = self.get_features(game_state, action)
        
        # Compute max Q-value for the next state
        legal_actions = next_state.get_legal_actions(self.index)
        if not legal_actions:
            max_q_next = 0.0
        else:
            max_q_next = max([self.get_q_value(next_state, a) for a in legal_actions])

        # Q-learning update rule
        current_q = self.get_q_value(game_state, action)
        difference = (reward + self.discount * max_q_next) - current_q
        
        # Check for potential explosion
        if abs(difference) > 10000:
             pass

        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None

        # Update weights if we are training
        if self.episodes_so_far < self.num_training:
            if self.last_state is not None:
                current_score = self.get_score(game_state)
                reward = current_score - self.last_score
                self.update(self.last_state, self.last_action, game_state, reward)
                self.last_score = current_score

        # Epsilon-greedy policy for training
        if self.episodes_so_far < self.num_training and util.flip_coin(self.epsilon):
            action = random.choice(actions)
        else:
            # Pick best action
            values = [self.get_q_value(game_state, a) for a in actions]
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            action = random.choice(best_actions)

        # Update last state/action
        self.last_state = game_state
        self.last_action = action
        
        return action

    def final(self, game_state):
        """
        Called at the end of each game.
        """
        # Final update
        if self.episodes_so_far < self.num_training:
            if self.last_state is not None:
                current_score = self.get_score(game_state)
                reward = current_score - self.last_score
                self.update(self.last_state, self.last_action, game_state, reward)
        
        self.episodes_so_far += 1
        if self.episodes_so_far == self.num_training:
            print(f"Training completed for agent {self.index}")
            print(f"Final Weights: {self.weights}")

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return self.weights


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def __init__(self, index, time_for_computing=.1, num_training=0):
        super().__init__(index, time_for_computing, num_training)
        # Initialize with some heuristic weights to speed up learning or start reasonable
        self.weights['successor_score'] = 100
        self.weights['distance_to_food'] = -1

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, time_for_computing=.1, num_training=0):
        super().__init__(index, time_for_computing, num_training)
        # Initialize with some heuristic weights
        self.weights['num_invaders'] = -1000
        self.weights['on_defense'] = 100
        self.weights['invader_distance'] = -10
        self.weights['stop'] = -100
        self.weights['reverse'] = -2

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
