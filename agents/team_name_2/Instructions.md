Agent Architecture

We use a hybrid team approach where the two agents think very differently:Offensive Agent (OffensiveQAgent):Type: Approximate Q-Learning Agent.Brain: It "Learns" from experience. It starts with no knowledge and gets smarter by playing games.Mechanism: It uses a weights.txt file to remember what strategies work. It calculates the value of an action using a weighted sum of features: $Q(s,a) = \sum (Weight_i \times Feature_i)$.Defensive Agent (DefensiveReflexAgent):Type: Reflex Agent.Brain: "Static Logic" (No Learning). It does not use weights.txt or training data.Mechanism: Its behavior is hard-coded with fixed weights (e.g., invader_distance: -10). It essentially says: "If I see an enemy, run toward them immediately." This ensures our defense is reliable from Game 1 and never "unlearns" its job.


2. Offensive Agent Features
The Offensive Agent uses a State Machine to switch behaviors dynamically. It doesn't just chase food; it decides when to chase and when to score.

A. Attack Mode (Default)
Active when we are safe and have low food.

successor_score: Counts food remaining. (Goal: Eat it).

distance_to_food: Distance to the nearest dot. (Goal: Minimize it).

distance_to_capsule: Distance to power pellets. (Goal: High Priority).

danger_ghost: Activates if a ghost is too close (< 4 steps). (Goal: Avoid!).

distance_to_ghost: Keeps a buffer zone from defenders.

B. Retreat Mode (Scoring)
Triggered if: We have 5+ dots, Time is low (<60 moves), or a Ghost is chasing us.

distance_to_home: Distance to the mid-line. (Goal: Run home to deposit points).

trapped: Checks if we are entering a dead-end. (Goal: Avoid getting cornered).

C. Interceptor Mode (Opportunity)
Triggered if: An enemy Pacman is detected on our side.

distance_to_invader: Even while attacking, if an enemy passes us, we will briefly switch to help kill them.

kill_invader: Bonus reward for eating the enemy.

3. How to Train the Model
Since the Offensive Agent learns, it needs "Training Camp" before the tournament.

Command:

Bash

python capture.py -r my_team.py -b baselineTeam.py -n 50 -q
-n 50: Runs 50 games in Training Mode.

The agent will explore random moves (epsilon) to find new strategies.

It will update weights.txt after every move.

-q: Runs in "Quiet Mode" (no graphics) so training finishes instantly.

Tournament Mode (No Training):

Bash

python capture.py -r my_team.py -b baselineTeam.py
Without -n, the agent stops learning (Exploration = 0). It strictly follows the best strategy stored in weights.txt.

4. The weights.txt File
This file acts as the agent's Long-Term Memory.

Input: At the start of a game, the agent reads this file to load its "Intelligence."

During Game: It multiplies these numbers by the features to make decisions.

Example: If distance_to_food has a weight of -10, the agent knows that being far from food is "bad."

Output: After training games, the agent saves its new, smarter weights back to this file.

Important: If this file is deleted, the agent gets "amnesia" and resets to being random.
