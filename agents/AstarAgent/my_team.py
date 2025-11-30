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
from game import Directions
from pathlib import Path


# ============================================================
# ====================   TEAM GENERATOR  =====================
# ============================================================

WEIGHTS_FILE = Path(__file__).parent / 'weights.txt'
NUM_TRAINING = 0
FINAL_TOURNAMENT_MODE = False


def create_team(first_index, second_index, is_red,
                first='SmartQLearningAgent', second='DefensiveAgent', num_training=0):
    """
    Factory to create the team for the framework.
    Keeps the original signature and behaviour.
    """
    global NUM_TRAINING
    NUM_TRAINING = num_training

    return [eval(first)(first_index), eval(second)(second_index)]


# ============================================================
# ====================   MAIN AGENT CLASS  ===================
# ============================================================

class SmartQLearningAgent(CaptureAgent):
    """
    Offensive Q-learning agent with:
    - Safety and retreat logic
    - Capsule prioritization
    - Optional defensive assistance through DefensiveAgent
    """

    # --------------------------------------------------------
    # Initialisation / setup
    # --------------------------------------------------------

    def register_initial_state(self, game_state):
        """
        Called by the framework once at the beginning.
        Sets up learning parameters, loads weights, and prepares helpers.
        """
        super().register_initial_state(game_state)

        self._init_learning_parameters()
        self._init_state_tracker()

        # Load weights (defaults only for stability)
        self.weights = None
        self.load_weights()

        # Snapshot of starting weights for safe_update clamping
        self.initial_weights = self.weights.copy()

        # Compute midline for home return logic
        self.compute_home_positions(game_state)

        # Helper defensive agent and mode flag
        self._init_defence_helper(game_state)

    def _init_learning_parameters(self):
        """Initialize Q-learning hyperparameters."""
        self.alpha = 1e-5
        self.discount = 0.9
        self.epsilon = 0.05
        self.num_training = NUM_TRAINING

    def _init_state_tracker(self):
        """Initialize tracking of last transition and simple stats."""
        self.last_state = None
        self.last_action = None
        self.last_score = 0

        # Track number of wins / losses during training runs
        self.wins = 0
        self.losses = 0

    def _init_defence_helper(self, game_state):
        """
        Create a local DefensiveAgent helper that this agent can
        temporarily delegate to when assisting on defence.
        """
        self.defence_helper = DefensiveAgent(self.index)
        self.defence_helper.red = self.red
        self.defence_helper.distancer = self.distancer
        self.defence_helper.register_initial_state(game_state)

        # Flag: are we currently helping on defence?
        self.assist_defence_mode = False

    # ============================================================
    # ======================  WEIGHTS LOGIC  =====================
    # ============================================================

    def load_weights(self):
        """Load weights from file if it exists, or fall back to defaults/tournament."""
        if WEIGHTS_FILE.exists() and WEIGHTS_FILE.stat().st_size > 0 and not FINAL_TOURNAMENT_MODE:
            try:
                with open(WEIGHTS_FILE, 'r') as f:
                    print("Loading weights from file.")
                    self.weights = eval(f.read())
            except Exception as e:
                print(f"Error loading weights: {e}")
                self.weights = self.default_weights()
        else:
            if FINAL_TOURNAMENT_MODE:
                print("Tournament mode: Using tournament weights.")
                self.weights = self.tournament_weights()
            else:
                print("Weights file not found or empty. Using default weights.")
                self.weights = self.default_weights()

    def final(self, game_state):
        """Called at the end of each game to log and optionally save weights."""
        print("Final weights:", self.weights)

        if self.get_score(game_state) > 0:
            self.wins += 1
        else:
            self.losses += 1

        print(f"Wins: {self.wins} out of {self.wins + self.losses} games.")

        # Only persist weights while in training mode
        if self.num_training > 0:
            try:
                with open(WEIGHTS_FILE, 'w') as f:
                    f.write(str(self.weights))
                    print("Weights saved to file.")
            except Exception as e:
                print(f"Error saving weights: {e}")

        CaptureAgent.final(self, game_state)

    # Handcrafted initial weights for features
    def default_weights(self):
        # Will be the starting point before training.
        return util.Counter({
            'bias': 1.0,
            'successor_score': 100.0,
            'distance_to_food': -2.0,
            'local_food_cluster': 5.0,    # strong preference for clustered food
            'ghost_proximity': -800.0,
            'stop': -100.0,
            'reverse': -3.0,
        })

    def tournament_weights(self):
        # Win rate 34/50 (these were trained externally)
        return util.Counter({
            'bias': 8.761410733666864,
            'successor_score': 90.0,
            'distance_to_food': -1.5572172875342591,
            'local_food_cluster': 14.722072774648437,
            'ghost_proximity': -800.0,
            'stop': -100.0,
            'reverse': -3.0,
        })

    # ============================================================
    # ======================  CHOOSE ACTION  =====================
    # ============================================================

    def choose_action(self, game_state):
        """
        Main decision function:
        1) Maybe switch to defensive helper.
        2) Capsule rescue logic.
        3) Emergency retreat.
        4) Strategic retreat.
        5) Q-learning action selection (ε-greedy).
        """
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
        POWER_MODE_TIMER_THRESHOLD = 2  # needs time to retreat after power ends
        power_mode = any(
            e.scared_timer > POWER_MODE_TIMER_THRESHOLD for e in defenders)

        # Distance to dangerous (non-scared) ghosts
        dangerous = [e for e in defenders if e.scared_timer == 0]
        min_ghost_dist = self.get_min_ghost_distance(
            game_state, my_pos, dangerous)

        # --------------------------------------------------------
        # 1) Optionally switch to defensive helper
        # --------------------------------------------------------
        action = self.offensive_to_defensive_Mode(game_state)
        if action is not None:
            self.record_transition(game_state, action)
            return action

        # --------------------------------------------------------
        # 2) Capsule as saviour?
        # --------------------------------------------------------
        action = self.capsular_nearby(game_state, my_pos, min_ghost_dist)
        if action is not None:
            self.record_transition(game_state, action)
            return action

        # --------------------------------------------------------
        # 3) Emergency retreat: enemy very close & not powered
        # --------------------------------------------------------
        if not power_mode and min_ghost_dist is not None and min_ghost_dist <= 3:
            safe_action = self.go_home_action(game_state)
            if safe_action is not None:
                self.record_transition(game_state, safe_action)
                return safe_action

        # --------------------------------------------------------
        # 4) Strategic retreat logic
        # --------------------------------------------------------
        action = self.strategic_retreat(game_state, power_mode,
                                        carrying, food_left, min_ghost_dist)
        if action is not None:
            self.record_transition(game_state, action)
            return action

        # --------------------------------------------------------
        # 5) Q-LEARNING ACTION SELECTION (ε-greedy)
        # --------------------------------------------------------
        legal_actions = game_state.get_legal_actions(self.index)

        # ε-greedy random exploration
        if self.num_training > 0 and random.random() < self.epsilon:
            action = random.choice(legal_actions)
            self.record_transition(game_state, action)
            return action

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

    # --------------------------------------------------------
    # Movement helpers
    # --------------------------------------------------------

    def go_toward_position_action(self, game_state, target_pos):
        """Choose the best move to move toward a specific target position."""
        legal = game_state.get_legal_actions(self.index)
        best_actions = []
        best_dist = float('inf')

        for a in legal:
            successor = self.get_successor(game_state, a)
            s_pos = nearest_point(
                successor.get_agent_state(self.index).get_position())
            d = self.safe_maze_distance(s_pos, target_pos)

            if d < best_dist:
                best_dist = d
                best_actions = [a]
            elif d == best_dist:
                best_actions.append(a)

        return random.choice(best_actions) if best_actions else None

    def safe_maze_distance(self, pos1, pos2):
        """
        Helper: maze distance with Manhattan fallback.
        Logic equivalent to try/except blocks used previously.
        """
        try:
            return self.get_maze_distance(pos1, pos2)
        except Exception:
            return util.manhattan_distance(pos1, pos2)

    # --------------------------------------------------------
    # Offensive ↔ Defensive mode logic
    # --------------------------------------------------------

    def offensive_to_defensive_Mode(self, game_state):
        """
        Decide whether this offensive agent should temporarily switch
        to defensive behaviour (using self.defence_helper).

        Returns:
            - An action (string) if we are in defence mode and let the
              helper choose the move.
            - None if we stay in normal offensive / Q-learning mode.
        """

        # Ensure the flag exists (in case of hot reload)
        if not hasattr(self, "assist_defence_mode"):
            self.assist_defence_mode = False

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # --- Basic game info ---
        score = self.get_score(game_state)
        losing = score < 0
        losing_by = -score if losing else 0

        # Time left (if the framework exposes it)
        time_left = None
        if hasattr(game_state, "data") and hasattr(game_state.data, "timeleft"):
            time_left = game_state.data.timeleft

        # Consider "late game" if little time is left
        late_game = (time_left is not None and time_left < 200)

        # Food near me (to measure offensive opportunity)
        food_positions = self.get_food(game_state).as_list()
        if food_positions:
            dist_to_closest_food = min(
                self.distancer.get_distance(my_pos, fpos)
                for fpos in food_positions
            )
            local_food_count = sum(
                1 for fpos in food_positions
                if self.distancer.get_distance(my_pos, fpos) <= 3
            )
        else:
            dist_to_closest_food = float("inf")
            local_food_count = 0

        # --- Enemy info ---
        enemy_indices = self.get_opponents(game_state)
        enemy_states = [game_state.get_agent_state(i) for i in enemy_indices]

        # Invaders: enemies on our side (Pacman) with known position
        invaders = [e for e in enemy_states
                    if e.is_pacman and e.get_position() is not None]

        # How much food they are carrying in total
        total_carried = sum(getattr(e, "num_carrying", 0) for e in invaders)

        # Distance to closest invader (if any)
        if invaders:
            dist_to_closest_invader = min(
                self.distancer.get_distance(my_pos, e.get_position())
                for e in invaders
            )
        else:
            dist_to_closest_invader = None

        # Thresholds
        CRITICAL_STOLEN_HARD = 10   # always panic if someone has this much
        MODERATE_STOLEN = 3         # moderate threat level

        # Invaders that currently carry a lot of our food
        dangerous_invaders = [
            e for e in invaders
            if getattr(e, "num_carrying", 0) >= CRITICAL_STOLEN_HARD
        ]

        # --- Update assist_defence_mode flag ---

        if self.assist_defence_mode:
            # Once we start helping, we STAY in defence mode until
            # there is NO invader carrying any food on our side.
            still_threat = any(getattr(e, "num_carrying", 0)
                               > 0 for e in invaders)
            if not still_threat:
                # Invader killed or scored / left our side
                self.assist_defence_mode = False

        else:
            # Not currently helping — decide if we should switch
            should_help = False

            # 1) Absolutely critical case: someone stole a lot
            if len(dangerous_invaders) > 0:
                should_help = True

            # 2) Moderate threat: invaders carrying some food & we are losing
            elif total_carried >= MODERATE_STOLEN and losing:
                # If late game or losing by a lot → more urgent
                if late_game or losing_by >= 5:
                    should_help = True
                else:
                    # If we are close to the invader and there isn't
                    # a ton of easy food around us, also help.
                    if (dist_to_closest_invader is not None and
                            dist_to_closest_invader <= 5 and
                            local_food_count < 3):
                        should_help = True

            # 3) Optional: time-pressure-only trigger:
            if (not should_help and late_game and total_carried > 0 and
                    losing_by >= 2):
                should_help = True

            if should_help:
                self.assist_defence_mode = True

        # --- If we are in assist defence mode, act like a defender ---
        if self.assist_defence_mode:
            # Let our helper DefensiveAgent choose the action
            self.defence_helper.observation_history = self.observation_history
            action = self.defence_helper.choose_action(game_state)
            return action

        # If we get here, we are NOT in defensive assist mode.
        # Caller should continue with normal offensive logic.
        return None

    # --------------------------------------------------------
    # Strategic retreat + capsule logic
    # --------------------------------------------------------

    def strategic_retreat(self, game_state, power_mode, carrying, food_left, min_ghost_dist):
        """
        Decide whether to retreat home based on:
        - Carried food
        - Remaining food
        - Ghost distance
        """
        # 1) Retreat carry threshold based on remaining food
        if food_left >= 20:
            retreat_carry_threshold = 3
        elif food_left >= 10:
            retreat_carry_threshold = 2
        else:
            retreat_carry_threshold = 1  # endgame safety

        # 2) Danger radius grows as agent carries more food
        danger_radius = 3 + min(carrying, 4)  # ranges 3 → 7

        # 3) Apply retreat rule ONLY when NOT powered
        retreat = (
            (not power_mode
             and carrying >= retreat_carry_threshold
             and min_ghost_dist is not None
             and min_ghost_dist <= danger_radius)
            or food_left == 0  # always retreat if no food left
        )

        if retreat:
            return self.go_home_action(game_state)
        return None

    def capsular_nearby(self, game_state, my_pos, min_ghost_dist):
        """
        Decide whether to go for a nearby capsule.

        Behaviour:
        - If a capsule is VERY close (<= ALWAYS_TAKE_DIST), always go eat it.
        - Else, if a ghost is close and the capsule is closer than the ghost,
          go for the capsule as a safety move.
        - Otherwise, do nothing (return None) and let normal logic handle food.

        Returns:
        - An action (string) if we decide to move towards a capsule.
        - None if we don't want to prioritize capsules this turn.
        """
        capsules = self.get_capsules(game_state)
        if not capsules:
            return None

        # --- Find closest capsule and its distance ---
        closest_cap = None
        min_cap_dist = None

        for c in capsules:
            d = self.safe_maze_distance(my_pos, c)

            if min_cap_dist is None or d < min_cap_dist:
                min_cap_dist = d
                closest_cap = c

        if closest_cap is None or min_cap_dist is None:
            return None

        # --- Rule 1: if capsule is very close, always take it ---
        ALWAYS_TAKE_DIST = 3
        if min_cap_dist <= ALWAYS_TAKE_DIST:
            return self.go_toward_position_action(game_state, closest_cap)

        # --- Rule 2: use capsule as an escape if ghost is close ---
        if min_ghost_dist is not None:
            DANGER_GHOST_DIST = 5
            if min_ghost_dist <= DANGER_GHOST_DIST and min_cap_dist < min_ghost_dist:
                return self.go_toward_position_action(game_state, closest_cap)

        # No capsule move chosen; caller should continue with normal logic
        return None

    # ============================================================
    # ==================  Q-LEARNING SUPPORT  ====================
    # ============================================================

    def record_transition(self, game_state, action):
        """
        Tracks state/action history and applies incremental Q-update.
        Ensures exactly one update per chosen action while training.
        """
        # Only if we are in training mode
        if self.last_state is not None and self.last_action is not None and self.num_training > 0:
            reward = self.get_score(game_state) - self.last_score
            self.safe_update(self.last_state, self.last_action,
                             game_state, reward)

        self.last_state = game_state
        self.last_action = action
        self.last_score = self.get_score(game_state)

    def safe_update(self, state, action, next_state, reward):
        """
        Numerically safe Q-learning update with:
        - Per-move weight change limit (step_per_move)
        - Global clamp around initial weights (max_total_shift)
        - Frozen "safety" weights that are not learned
        """

        # Ensure we have an initial snapshot of the starting weights
        if not hasattr(self, "initial_weights"):
            self.initial_weights = self.weights.copy()

        # --- 1. Compute V(s') = max_a' Q(s', a') ---
        next_actions = next_state.get_legal_actions(self.index)

        q_next_values = []
        for a in next_actions:
            q_val = self.evaluate(next_state, a)
            if math.isfinite(q_val):
                q_next_values.append(q_val)

        # Terminal state: no legal actions => value 0
        max_next_q = max(q_next_values) if q_next_values else 0.0

        # --- 2. Current Q(s, a) ---
        current_q = self.evaluate(state, action)
        if not math.isfinite(current_q):
            current_q = 0.0

        # --- 3. TD error δ = [R + γ max_a' Q(s', a')] - Q(s, a) ---
        target = reward + self.discount * max_next_q
        difference = target - current_q
        if not math.isfinite(difference):
            return  # skip bad numerics

        # --- 4. Weight update with safety constraints ---

        # Weights we do NOT want to change (keep strong safety behaviour)
        FROZEN_WEIGHTS = {"ghost_proximity", "stop", "reverse"}

        # Weights that must stay <= 0
        CONSTRAINED_NEGATIVE = {"distance_to_food"}

        # Max change per *move* (per call to safe_update) for any single weight
        step_per_move = 0.1

        # Max total drift allowed from the initial (hand-crafted / loaded) weight
        max_total_shift = 10.0

        features = self.get_features(state, action)

        for f, v in features.items():
            # Skip frozen weights entirely
            if f in FROZEN_WEIGHTS:
                continue

            # Skip non-finite feature values
            if not math.isfinite(v):
                continue

            # Standard linear Q-learning update term: α * δ * feature_value
            raw_update = self.alpha * difference * v
            if not math.isfinite(raw_update):
                continue

            # --- Per-move clipping: limit the *update* size ---
            if raw_update > step_per_move:
                raw_update = step_per_move
            elif raw_update < -step_per_move:
                raw_update = -step_per_move

            # --- Global clamp around initial weight ---
            base_w = self.initial_weights.get(f, 0.0)
            current_w = self.weights.get(f, base_w)

            proposed = current_w + raw_update

            lower_global = base_w - max_total_shift
            upper_global = base_w + max_total_shift

            if proposed > upper_global:
                new_weight = upper_global
            elif proposed < lower_global:
                new_weight = lower_global
            else:
                new_weight = proposed

            # Enforce negativity for certain features
            if f in CONSTRAINED_NEGATIVE and new_weight > 0.0:
                new_weight = -2.0  # small negative value

            self.weights[f] = new_weight

    # ============================================================
    # ====================  FEATURE EXTRACTOR  ===================
    # ============================================================

    def get_features(self, game_state, action):
        """
        Feature extractor for Q(s,a):
        - bias
        - successor_score (remaining food)
        - distance_to_food
        - local_food_cluster
        - ghost_proximity
        - stop / reverse penalties
        """
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
                d = self.safe_maze_distance(my_pos, f)

                if min_dist is None or d < min_dist:
                    min_dist = d
                    closest = f

            features['distance_to_food'] = min_dist

            # Count cluster neighbors near closest food
            cx, cy = closest
            cluster = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (cx + dx, cy + dy) in food_list:
                    cluster += 1
            features['local_food_cluster'] = cluster

        # Ghost proximity
        defenders = [s for s in self.get_opponents_states(successor)
                     if not s.is_pacman and s.get_position()]

        dangerous = [s for s in defenders if s.scared_timer == 0]
        min_gdist = self.get_min_ghost_distance(successor, my_pos, dangerous)

        if min_gdist is not None:
            # FIXED LOGIC: closer ghosts → larger positive penalty,
            # combined with a negative weight in default/tournament weights.
            if min_gdist < 2:
                features['ghost_proximity'] = 1000
            elif min_gdist < 5:
                features['ghost_proximity'] = 200 * (5 - min_gdist)

        # Stop penalty
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
    # =====================  GHOST DISTANCE  =====================
    # ============================================================

    def get_min_ghost_distance(self, game_state, my_pos, ghosts):
        """Return the minimum distance to any ghost in the given list."""
        dists = []
        for g in ghosts:
            d = self.safe_maze_distance(
                my_pos, nearest_point(g.get_position()))
            dists.append(d)
        return min(dists) if dists else None

    def get_opponents_states(self, game_state):
        return [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]

    # ============================================================
    # ======================  GO HOME LOGIC  =====================
    # ============================================================

    def compute_home_positions(self, game_state):
        """Compute set of boundary squares used to retreat home."""
        walls = game_state.get_walls()
        mid = walls.width // 2 - (1 if self.red else 0)

        self.home_positions = []
        for y in range(walls.height):
            if not walls[mid][y] and not walls[mid - 1][y]:
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
                d = self.safe_maze_distance(s_pos, home)

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
    """
    Simple defensive agent:
    - Patrols along midline
    - Chases visible invaders
    - Avoids invaders when scared
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        # Precompute midline patrol waypoints
        self.patrol_points = self.compute_patrol_points(game_state)
        self.patrol_index = 0

    ###########################################################
    # ===================== CHOOSE ACTION =====================
    ###########################################################

    def choose_action(self, state):
        my_state = state.get_agent_state(self.index)
        my_pos = nearest_point(my_state.get_position())

        enemies = [state.get_agent_state(i) for i in self.get_opponents(state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]

        scared = my_state.scared_timer > 2

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
    # ===================== DEFENSIVE LOGIC ===================
    ###########################################################

    def chase_invader(self, state, my_pos, invaders):
        legal = state.get_legal_actions(self.index)

        # Closest invader
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
    # ========================== PATROL =======================
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
    # ========================== UTILITIES ====================
    ###########################################################

    def compute_patrol_points(self, state):
        """
        Option 3: fixed midline waypoints.
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
        """Maze distance with Manhattan fallback."""
        try:
            return self.get_maze_distance(a, b)
        except Exception:
            return util.manhattan_distance(a, b)
