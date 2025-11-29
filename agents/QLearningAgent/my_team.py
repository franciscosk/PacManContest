from contest.capture_agents import CaptureAgent
import random
import time
import contest.util as util
from contest.game import Directions, Actions
import contest.game as game
from contest.util import nearest_point

#################
# Team creation #
#################

NUM_TRAINING = 0
TRAINING = False
####

# COMMAND


# python capture.py -r ../../agents/QLearningAgent/my_team.py -b baseline_team.py  -n 20  --delay-step 0 -l RANDOM


####

def create_team(first_index, second_index, is_red,
                first='ApproxQLearningOffense', second='DefensiveReflexAgent', num_training=0, **args):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    """

    # The following line is an example only; feel free to change it.
    global NUM_TRAINING
    NUM_TRAINING = num_training
    return [eval(first)(first_index), eval(second)(second_index)]


class ApproxQLearningOffense(CaptureAgent):

    def register_initial_state(self, game_state):
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.num_training = NUM_TRAINING
        self.episodes_so_far = 0

        self.weights = {'closest-food': -3.099192562140742,
                        'bias': -9.280875042529367,
                        '#-of-ghosts-1-step-away': -16.6612110039328,
                        'eats-food': 11.127808437648863}

        self.start = game_state.get_agent_position(self.index)
        self.features_extractor = FeaturesExtractor(self)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
            Picks among the actions with the highest Q(s,a).
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        action = None
        if TRAINING:
            for action in legal_actions:
                self.update_weights(game_state, action)
        if not util.flip_coin(self.epsilon):
            # exploit
            action = self.get_policy(game_state)
        else:
            # explore
            action = random.choice(legal_actions)
        return action

    def get_weights(self):
        return self.weights

    def get_q_value(self, game_state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # features vector
        features = self.features_extractor.get_features(game_state, action)
        return features * self.weights

    def update(self, game_state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        features = self.features_extractor.get_features(game_state, action)
        old_value = self.get_q_value(game_state, action)
        future_q_value = self.get_value(next_state)
        difference = (reward + self.discount * future_q_value) - old_value
        # for each feature i
        for feature in features:
            new_weight = self.alpha * difference * features[feature]
            self.weights[feature] += new_weight
        # print(self.weights)

    def update_weights(self, game_state, action):
        next_state = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, next_state)
        self.update(game_state, action, next_state, reward)

    def get_reward(self, game_state, next_state):
        reward = 0
        agent_position = game_state.get_agent_position(self.index)

        # check if I have updated the score
        if self.get_score(next_state) > self.get_score(game_state):
            diff = self.get_score(next_state) - self.get_score(game_state)
            reward = diff * 10

        # check if food eaten in next_state
        my_foods = self.get_food(game_state).as_list()
        dist_to_food = min([self.get_maze_distance(agent_position, food)
                            for food in my_foods]) if my_foods else 9999
        # I am 1 step away, will I be able to eat it?
        if dist_to_food == 1:
            next_foods = self.get_food(next_state).as_list()
            if len(my_foods) - len(next_foods) == 1:
                reward = 10

        # check if I am eaten
        enemies = [game_state.get_agent_state(i)
                   for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position()
                  != None]
        if len(ghosts) > 0:
            min_dist_ghost = min([self.get_maze_distance(
                agent_position, g.get_position()) for g in ghosts])
            if min_dist_ghost == 1:
                next_pos = next_state.get_agent_state(
                    self.index).get_position()
                if next_pos == self.start:
                    # I die in the next state
                    reward = -100

        return reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        CaptureAgent.final(self, state)
        # print(self.weights)
        # did we finish training?

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

    def compute_value_from_q_values(self, game_state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        allowed_actions = game_state.get_legal_actions(self.index)
        if len(allowed_actions) == 0:
            return 0.0
        best_action = self.get_policy(game_state)
        return self.get_q_value(game_state, best_action)

    def compute_action_from_q_values(self, game_state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None
        action_vals = {}
        best_q_value = float('-inf')
        for action in legal_actions:
            target_q_value = self.get_q_value(game_state, action)
            action_vals[action] = target_q_value
            if target_q_value > best_q_value:
                best_q_value = target_q_value
        best_actions = [k for k, v in action_vals.items() if v == best_q_value]
        # random tie-breaking
        return random.choice(best_actions)

    def get_policy(self, game_state):
        return self.compute_action_from_q_values(game_state)

    def get_value(self, game_state):
        return self.compute_value_from_q_values(game_state)


class FeaturesExtractor:

    def __init__(self, agent_instance):
        self.agent_instance = agent_instance

    def get_features(self, game_state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = self.agent_instance.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(
            i) for i in self.agent_instance.get_opponents(game_state)]
        ghosts = [a.get_position()
                  for a in enemies if not a.is_pacman and a.get_position() != None]
        # ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        agent_position = game_state.get_agent_position(
            self.agent_instance.index)
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        # if len(ghosts) > 0:
        #   minGhostDistance = min([self.agent_instance.get_maze_distance(agent_position, g) for g in ghosts])
        #   if minGhostDistance < 3:
        #     features["minGhostDistance"] = minGhostDistance

        # successor = self.agent_instance.get_successor(game_state, action)
        # features['successorScore'] = self.agent_instance.get_score(successor)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # capsules = self.agent_instance.get_capsules(game_state)
        # if len(capsules) > 0:
        #   closestCap = min([self.agent_instance.get_maze_distance(agent_position, cap) for cap in self.agent_instance.get_capsules(game_state)])
        #   features["closestCapsule"] = closestCap

        dist = self.closest_food((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / \
                (walls.width * walls.height)
        features.divide_all(10.0)
        # print(features)
        return features

    def closest_food(self, pos, food, walls):
        """
        closest_food -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

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

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successorScore'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if my_state.is_pacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
