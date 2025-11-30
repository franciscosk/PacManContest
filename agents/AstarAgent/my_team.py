# my_team.py
# Clean, stable Q-learning Capture agent with:
# - NaN-safe updates
# - Capsule ("white thing") logic
# - Retreat when enemy approaches
# - Clustered food attraction
# - Clean architecture and modular design

from capture_agents import CaptureAgent
import util
from util import nearest_point
import math
import random
from game import Directions   # <-- ADD THIS IMPORT


# ============================================================
# ====================   TEAM GENERATOR  =====================
# ============================================================

def create_team(first_index, second_index, is_red,
                first='SmartQLearningAgent', second='DefensiveAgent', num_training=0):

    return [eval(first)(first_index), eval(second)(second_index)]


# ============================================================
# ====================   MAIN AGENT CLASS  ====================
# ============================================================

class SmartQLearningAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Q-learning parameters
        self.alpha = 0.02
        self.discount = 0.9
        self.epsilon = 0.05
        self.num_training = 0

        # State tracking
        self.last_state = None
        self.last_action = None
        self.last_score = 0

        # Load weights (defaults only for stability)
        self.weights = self.default_weights()

        # Compute midline for home return logic
        self.compute_home_positions(game_state)

    # ============================================================
    # ======================  DEFAULT WEIGHTS  ====================
    # ============================================================

    def default_weights(self):
        return util.Counter({
            'bias': 1.0,
            'successor_score': 100.0,
            'distance_to_food': -2.0,
            'local_food_cluster': 5.0,    # strong preference for clustered food
            'ghost_proximity': -800.0,
            'stop': -100.0,
            'reverse': -3.0,
            # 'distance_to_capsule': -5.0,   # stronger incentive to move toward it
            # 'capsule_in_range': 200.0      # very high immediate value
        })

    # ============================================================
    # ======================  CHOOSE ACTION  ======================
    # ============================================================

    def choose_action(self, game_state):
        # --------------------------------------------------------
        # Extract useful state info
        # --------------------------------------------------------
        my_state = game_state.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())
        carrying = my_state.num_carrying
        food_list = self.get_food(game_state).as_list()
        food_left = len(food_list)

        enemies = [game_state.get_agent_state(
            i) for i in self.get_opponents(game_state)]
        defenders = [e for e in enemies if (
            not e.is_pacman) and e.get_position() is not None]

        # Check capsule power mode (scared ghosts)
        power_mode = any(e.scared_timer > 0 for e in defenders)

        # Distance to dangerous (non-scared) ghosts
        dangerous = [e for e in defenders if e.scared_timer == 0]
        min_ghost_dist = self.get_min_ghost_distance(
            game_state, my_pos, dangerous)

        # ====================== Capsula as savior? ======================

        # --------------------------------------------------------
        # Emergency retreat: enemy very close & not powered
        # --------------------------------------------------------
        if not power_mode and min_ghost_dist is not None and min_ghost_dist <= 3:
            safe_action = self.go_home_action(game_state)
            if safe_action is not None:
                self.record_transition(game_state, safe_action)
                return safe_action

        # --------------------------------------------------------
        # Standard retreat: carrying 3+ food or little food left
        # BUT skip retreat if capsule (white thing) was eaten
        # --------------------------------------------------------
        # if not power_mode and (carrying >= 3 or food_left <= 2):
        #     safe_action = self.go_home_action(game_state)
        #     if safe_action is not None:
        #         self.record_transition(game_state, safe_action)
        #         return safe_action

         # 1) Retreat carry threshold based on remaining food
        if food_left >= 20:
            retreat_carry_threshold = 3
        elif food_left >= 10:
            retreat_carry_threshold = 2
        else:
            retreat_carry_threshold = 1  # endgame safety

        # 2) Danger radius grows as agent carries more food
        danger_radius = 3 + min(carrying, 2)  # ranges 3 → 5

        # 3) Apply retreat rule ONLY when NOT powered
        retreat = (
            not power_mode
            and carrying >= retreat_carry_threshold
            and min_ghost_dist is not None
            and min_ghost_dist <= danger_radius
        )

        if retreat:
            action = self.go_home_action(game_state)
            self.record_transition(game_state, action)
            return action

        # --------------------------------------------------------
        # Q-LEARNING ACTION SELECTION
        # --------------------------------------------------------
        legal_actions = game_state.get_legal_actions(self.index)

        # ε-greedy random exploration
        if self.num_training > 0 and random.random() < self.epsilon:
            a = random.choice(legal_actions)
            self.record_transition(game_state, a)
            return a

        # Exploitation: choose best Q-value
        best_actions = []
        best_value = float('-inf')
        for a in legal_actions:
            q = self.evaluate(game_state, a)
            if q > best_value:
                best_value = q
                best_actions = [a]
            elif q == best_value:
                best_actions.append(a)

        chosen = random.choice(best_actions)
        self.record_transition(game_state, chosen)
        return chosen

    # ============================================================
    # ==================  Q-LEARNING SUPPORT  ====================
    # ============================================================

    def record_transition(self, game_state, action):
        """Tracks state/action history and applies incremental Q-update."""
        if self.last_state is not None and self.last_action is not None and self.num_training > 0:
            reward = self.get_score(game_state) - self.last_score
            self.safe_update(self.last_state, self.last_action,
                             game_state, reward)
        self.last_state = game_state
        self.last_action = action
        self.last_score = self.get_score(game_state)

    # ------------------------ SAFE UPDATE ------------------------

    def safe_update(self, state, action, next_state, reward):
        """Numerically safe Q-learning update (never produces NaN)."""

        # Compute max_a' Q(s', a')
        next_actions = next_state.get_legal_actions(self.index)
        q_next_values = []
        for a in next_actions:
            val = self.evaluate(next_state, a)
            if math.isfinite(val):
                q_next_values.append(val)

        max_next_q = max(q_next_values) if q_next_values else 0.0

        # Current Q(s,a)
        current_q = self.evaluate(state, action)
        if not math.isfinite(current_q):
            current_q = 0.0

        # TD target
        target = reward + self.discount * max_next_q
        difference = target - current_q

        if not math.isfinite(difference):
            return  # Skip bad update

        # Update weights
        features = self.get_features(state, action)
        for f, v in features.items():
            if math.isfinite(v):
                self.weights[f] += self.alpha * difference * v

    # ============================================================
    # ====================  FEATURE EXTRACTOR  ====================
    # ============================================================

    def get_features(self, game_state, action):
        features = util.Counter()

        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        features['bias'] = 1.0

        # Food-related features
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        if food_list:
            # Find closest food
            closest, min_dist = None, None
            for f in food_list:
                try:
                    d = self.get_maze_distance(my_pos, f)
                except:
                    d = util.manhattan_distance(my_pos, f)

                if min_dist is None or d < min_dist:
                    min_dist = d
                    closest = f

            features['distance_to_food'] = min_dist

            # Count cluster neighbors near closest food
            cx, cy = closest
            cluster = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (cx+dx, cy+dy) in food_list:
                    cluster += 1
            features['local_food_cluster'] = cluster

         # Capsule-related features
        # capsules = self.get_capsules(successor)
        # if capsules:
        #     # Find closest capsule
        #     closest_cap, min_cap_dist = None, None
        #     for c in capsules:
        #         try:
        #             d = self.get_maze_distance(my_pos, c)
        #         except:
        #             d = util.manhattan_distance(my_pos, c)

        #         if min_cap_dist is None or d < min_cap_dist:
        #             min_cap_dist = d
        #             closest_cap = c

        #     features['distance_to_capsule'] = min_cap_dist

        #     # If successor position is exactly on capsule → reward heavily
        #     if my_pos == closest_cap:
        #         features['capsule_in_range'] = 1.0
        #     else:
        #         features['capsule_in_range'] = 0.0
        # else:
        #     features['distance_to_capsule'] = 0.0
        #     features['capsule_in_range'] = 0.0

        # Ghost proximity
        defenders = [s for s in self.get_opponents_states(successor)
                     if not s.is_pacman and s.get_position()]

        dangerous = [s for s in defenders if s.scared_timer == 0]
        min_gdist = self.get_min_ghost_distance(successor, my_pos, dangerous)

        if min_gdist is not None:
            if min_gdist < 2:
                features['ghost_proximity'] = -1000
            elif min_gdist < 5:
                features['ghost_proximity'] = -200 * (5 - min_gdist)

        # Stop / reverse penalties
      # Stop / reverse penalties
        if action == 'Stop':
            features['stop'] = 1.0

        # Reverse direction penalty (using Directions.REVERSE)
        current_dir = game_state.get_agent_state(
            self.index).configuration.direction
        if current_dir in Directions.REVERSE:
            rev = Directions.REVERSE[current_dir]
            if action == rev:
                features['reverse'] = 1.0

        return features

    # ============================================================
    # ==================  Q-VALUE AND SUCCESSORS  =================
    # ============================================================

    def evaluate(self, state, action):
        """Compute Q(s,a) = w·f safely."""
        features = self.get_features(state, action)
        return sum(self.weights[f] * features[f] for f in features)

    def get_successor(self, state, action):
        """Handles partial movement via nearest_point."""
        successor = state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            successor = successor.generate_successor(self.index, action)
        return successor

    # ============================================================
    # =====================  GHOST DISTANCE  ======================
    # ============================================================

    def get_min_ghost_distance(self, game_state, my_pos, ghosts):
        dists = []
        for g in ghosts:
            try:
                d = self.get_maze_distance(
                    my_pos, nearest_point(g.get_position()))
            except:
                d = util.manhattan_distance(
                    my_pos, nearest_point(g.get_position()))
            dists.append(d)
        return min(dists) if dists else None

    def get_opponents_states(self, game_state):
        return [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

    # ============================================================
    # ======================  GO HOME LOGIC  ======================
    # ============================================================

    def compute_home_positions(self, game_state):
        """Compute set of boundary squares used to retreat home."""
        walls = game_state.get_walls()
        mid = walls.width // 2 - (1 if self.red else 0)

        self.home_positions = []
        for y in range(walls.height):
            if not walls[mid][y] and not walls[mid-1][y]:
                self.home_positions.append((mid, y))

    def go_home_action(self, game_state):
        """Choose the best move to move toward the nearest home boundary."""
        my_state = game_state.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())
        legal = game_state.get_legal_actions(self.index)

        best_actions = []
        best_dist = float('inf')

        for a in legal:
            successor = self.get_successor(game_state, a)
            s_pos = nearest_point(
                successor.get_agent_state(self.index).get_position())

            for home in self.home_positions:
                try:
                    d = self.get_maze_distance(s_pos, home)
                except:
                    d = util.manhattan_distance(s_pos, home)

                if d < best_dist:
                    best_dist = d
                    best_actions = [a]
                elif d == best_dist:
                    best_actions.append(a)

        return random.choice(best_actions) if best_actions else None

 ###############################################################
# ======================== DEFENSIVE AGENT ====================
###############################################################


class DefensiveAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Precompute midline patrol waypoints (Option 3)
        self.patrol_points = self.compute_patrol_points(game_state)
        self.patrol_index = 0

    ###########################################################
    # ===================== CHOOSE ACTION ======================
    ###########################################################

    def choose_action(self, state):
        my_state = state.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        enemies = [state.get_agent_state(i) for i in self.get_opponents(state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]

        scared = my_state.scared_timer > 0

        # ------------------------------------------------------
        # SCARED: Avoid invaders
        # ------------------------------------------------------
        if scared and invaders:
            return self.avoid_invaders(state, my_pos, invaders)

        # ------------------------------------------------------
        # CHASE MODE: Invader detected
        # ------------------------------------------------------
        if invaders:
            return self.chase_invader(state, my_pos, invaders)

        # ------------------------------------------------------
        # PATROL MODE (no invaders)
        # ------------------------------------------------------
        return self.patrol(state, my_pos)

    ###########################################################
    # ===================== DEFENSIVE LOGIC ====================
    ###########################################################

    def chase_invader(self, state, my_pos, invaders):
        legal = state.get_legal_actions(self.index)

        # closest invader
        target = min(invaders, key=lambda e: self.safe_dist(
            my_pos, e.get_position(), state))
        tpos = nearest_point(target.get_position())

        best = []
        best_dist = float("inf")

        for a in legal:
            succ = state.generate_successor(self.index, a)
            s_pos = succ.get_agent_state(self.index).get_position()
            if s_pos is None:
                continue
            s_pos = nearest_point(s_pos)

            d = self.safe_dist(s_pos, tpos, state)
            if d < best_dist:
                best_dist = d
                best = [a]
            elif d == best_dist:
                best.append(a)

        return random.choice(best) if best else random.choice(legal)

    def avoid_invaders(self, state, my_pos, invaders):
        legal = state.get_legal_actions(self.index)

        target = min(invaders, key=lambda e: self.safe_dist(
            my_pos, e.get_position(), state))
        tpos = nearest_point(target.get_position())

        # Maximize distance from invader
        best = []
        best_dist = float("-inf")

        for a in legal:
            succ = state.generate_successor(self.index, a)
            s_pos = succ.get_agent_state(self.index).get_position()
            if not s_pos:
                continue
            s_pos = nearest_point(s_pos)

            d = self.safe_dist(s_pos, tpos, state)
            if d > best_dist:
                best_dist = d
                best = [a]
            elif d == best_dist:
                best.append(a)

        return random.choice(best) if best else random.choice(legal)

    ###########################################################
    # ========================== PATROL =========================
    ###########################################################

    def patrol(self, state, my_pos):
        legal = state.get_legal_actions(self.index)

        # Choose next patrol point
        target = self.patrol_points[self.patrol_index]

        if my_pos == target:
            # Move to next patrol target
            self.patrol_index = (self.patrol_index +
                                 1) % len(self.patrol_points)
            target = self.patrol_points[self.patrol_index]

        # Move toward patrol point
        best_actions = []
        best_dist = float("inf")

        for a in legal:
            succ = state.generate_successor(self.index, a)
            s_pos = succ.get_agent_state(self.index).get_position()
            if not s_pos:
                continue

            s_pos = nearest_point(s_pos)
            d = self.safe_dist(s_pos, target, state)

            if d < best_dist:
                best_dist = d
                best_actions = [a]
            elif d == best_dist:
                best_actions.append(a)

        return random.choice(best_actions) if best_actions else random.choice(legal)

    ###########################################################
    # ========================== UTILITIES ======================
    ###########################################################

    def compute_patrol_points(self, state):
        """
        Option 3: fixed midline waypoints
        """
        walls = state.get_walls()
        mid = walls.width // 2 - (1 if self.red else 0)

        points = []
        height = walls.height

        # three evenly spaced waypoints
        for y in [height // 4, height // 2, 3 * height // 4]:
            if not walls[mid][y]:
                points.append((mid, y))

        # fallback in case walls block our ideal positions
        if not points:
            for y in range(1, height - 1):
                if not walls[mid][y]:
                    points.append((mid, y))
                    break

        return points

    def safe_dist(self, a, b, s):
        try:
            return self.get_maze_distance(a, b)
        except:
            return util.manhattan_distance(a, b)
