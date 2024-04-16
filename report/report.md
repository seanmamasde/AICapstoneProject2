---
title: "Artificial Intelligence Capstone: Homework 2"
author:
  - Team No. 19
  - 110550032 文玠敦
  - 110550038 莊書杰
  - 110550074 馬楷翔
date: \today{}
CJKmainfont: "Microsoft YaHei"
---

## Introduction

In this assignment, we designed game agents for a board game called **Battle Sheep**.
This project is divided into 4 parts, each part is a slightly different variant of the original board game (most of the
rules remain unchanged), and presents a different set of criteria as challenges for our agent, which is as follows:

1. The Basic Form
   The game is conducted on a `12*12` board, which then will randomly generate a connected portion consists of `64`
   playable cells.
   Each of the 4 players gets `16` sheep initially.
2. Overcrowded Bigger Board
   Each player now has `32` sheep to begin the game with.
   Also, the field is now `100` cells in total chosen within
   a `15*15` board.
3. Imperfect Information
   A player (which is out agent) does not know the number of sheep in cells occupied by other players.
4. Cooperative Play
   The 4 players form 2 teams: with player 1 & 3 on the same team and 2 & 4 on the other.

## Methodology

We have implemented `Monte Carlo Tree Search (MCTS)` algorithm as our agent decision-making strategy.

`Monte Carlo Tree Search (MCTS)` is an algorithm used for decision tree search, particularly well-suited for situations
where the state space is large and exhaustive search is impractical, such as in games like Go.
It's a heuristic search algorithm that evaluates possible actions by simulating random games, gradually building a
search tree, and guiding the search based on simulation results.
The algorithm includes 4 parts:

1. Selection
   Starting from the root node, the algorithm selects child nodes based on a certain strategy (such as UCB applied to
   trees, UCT) until it finds an unexplored node.
2. Expansion
   Starting from the current game state, the algorithm expands the search tree by simulating the game according to a
   certain strategy (usually random selection) until a certain condition is met
   (such as reaching a leaf node or maximum depth).
3. Simulation
   The algorithm simulates the game starting from the selected node,
   often using random actions or heuristic strategies, until it reaches a terminal game state.
4. Back Propagation
   Based on the result of the simulation, the algorithm updates statistical information (such as win counts and visit
   counts) for all nodes along the search path.
   This information guides future node selections.

Repeat the above steps until a predetermined search time, iteration count, or another termination condition is met.

![](https://hackmd.io/_uploads/BkGfU3clA.png)

## Implementation

### Game 1

Parameters

- `BOARD_SIZE` = 12
- `SHEEP_NUM` = 16
- `TIME_LIMIT` = 3 * 0.95
- `final_score` = [1, 0.5, 0.3, 0.1]

In our implementation, each node contains the following information:

- The current player to be controlled.
- The action taken by the previous player.
- The current `mapStat` and `sheepStat`.
- The number of visits to this node.
- The score of this node.
- Whether this node represents the first action of the player.

During each iteration, the following operations are performed:

1. Selection
   First, calculate the UCT score based on the node's information, and choose the child node with the highest UCT score
   until reaching a leaf node.
2. Expansion
   Creates child nodes for all possible actions at the current state and returns the total number of possible actions.
   For the player's first action, choose any border cell to place sheep.
   For later actions, select the divided group, the number of divisions, and the position of movement.
3. Simulation
   If the node has no available actions, let each player take random actions from that point until the game ends, and
   return the score obtained based on the game result.
   If the node has available actions, randomly choose one child node and let each player take random actions until the
   game ends, then return the score obtained based on the game result.
4. Back Propagation
   Update the score and visit count of all nodes along the path from the root node to the leaf node.
   Noted that the score to be updated for each node should be the score of the player from the parent node, as the child
   node represents the outcome obtained after the parent node makes a decision.

When the search reaches our predefined time limit, select the node with the highest number of visits among the children
of the root node and take action accordingly.

### Game 2

Parameters

- `BOARD_SIZE` = 15
- `SHEEP_NUM` = 32
- `TIME_LIMIT` = 3 * 0.95
- `final_score` = [1, 0.5, 0.3, 0.1]

Most implementations are similar to game 1, but with adjustments made to certain parameters.

### Game 3

Parameters

- `BOARD_SIZE` = 12
- `SHEEP_NUM` = 16
- `TIME_LIMIT` = 3 * 0.95
- `final_score` = [1, 0.5, 0.3, 0.1]

Most implementations are similar to game 1, but since we need to address the issue of not being able to see the
opponent's sheep count, we have evenly distributed the opponent's sheep among each flock.

### Game 4

Parameters

- `BOARD_SIZE` = 12
- `SHEEP_NUM` = 16
- `TIME_LIMIT` = 3 * 0.95
- `final_score` = [1, 0.7, 0.2, 0.1]

In game 4, we've changed the algorithm a bit to meet the **team** requirement.
As opposed to the original scoring criteria, the modified one focuses more on the team factor of this game:

```python!
# adjust scores by combining player 1 and player 3, and treating them as a team
team_score = scores[0] + scores[2]
opponent_score = scores[1] + scores[3]
```

In addition, we've also adjusted the `Simulate` function accordingly, in order to reflect the team situation more
accurately.

These modifications enable our game agent to put the team criteria into its consideration when making decisions.

## Experiments

### Original Setting

In the original setting, we set the `final_score` variable as `[1.0, 0.5, 0.3, 0.1]`, and this is the result:

| iteration | player 19 | player 2 | player 3 | player 4 | winner |
|:---------:|-----------|----------|----------|----------|:------:|
|     1     | 30.51985  | 8.0      | 14.1336  | 12.03527 |   19   |
|     2     | 24.68478  | 10.70505 | 13.32664 | 11.85516 |   19   |
|     3     | 17.04736  | 26.68478 | 11.03527 | 17.45434 |   2    |
|     4     | 23.33452  | 16.33873 | 10.32664 | 13.13524 |   19   |
|     5     | 24.68478  | 12.47674 | 11.65685 | 15.33873 |   19   |
|     6     | 22.93109  | 18.04289 | 12.76892 | 13.03527 |   19   |
|     7     | 15.32664  | 7.65685  | 7.37841  | 14.38604 |   19   |
|     8     | 26.3594   | 22.61034 | 8.75683  | 22.41118 |   19   |
|     9     | 24.68478  | 14.03527 | 16.41368 | 12.94822 |   19   |
|    10     | 23.33452  | 12.98349 | 12.60508 | 15.58846 |   19   |

We ran `10` iterations in total, and the resulting win rate is about `90%`, with an average score of `23.29077`, higher
than all 3 other players.

### Extreme Value for First Place

In this experiment the `final_score` array is modified to `[5, -0.5, -1, -2]`, and the result is as follows:

| iteration | player 19 | player 2 | player 3 | player 4 | winner |
|:---------:|-----------|----------|----------|----------|:------:|
|     1     | 28.08071  | 12.37841 | 16.83276 | 14.60508 |   19   |
|     2     | 29.51985  | 12.76892 | 20.42578 | 11.38604 |   19   |
|     3     | 25.68962  | 20.96687 | 9.32664  | 17.45434 |   19   |
|     4     | 25.71293  | 23.41118 | 17.40256 | 14.45434 |   19   |
|     5     | 27.68478  | 18.83276 | 18.83276 | 13.32664 |   19   |
|     6     | 25.33452  | 17.14734 | 13.45434 | 17.27486 |   19   |
|     7     | 32        | 18.96687 | 16.1336  | 15.71714 |   19   |
|     8     | 27.68478  | 7.89644  | 18.78279 | 23.91509 |   19   |
|     9     | 24.03276  | 18.78279 | 13.27486 | 14.60508 |   19   |
|    10     | 22.58846  | 16.45434 | 12.38604 | 2.37841  |   19   |

We ran `10` iterations for this part, and the result is much better compared to the original setting, with an average
score of `26.83284` and around `100%` in win rate.
In addition, it even got one full mark, which is pretty impressive.

### Minimax Method

#### Setting

In addition to `MCTS`, we have also tried using Minimax method to implement game 1~4. We define a class Node which
represents a state in the game tree.

Each node contains the following information:

- The current player to be controlled.
- The action taken by the previous player.
- The current `mapStat` and `sheepStat`.
- The score of this node.

Similar to `MCTS`, we looped through and simulated all available starting cells by applying the
function `generate_children`, and selected the cell that has the highest score to be our initial position.

Starting from the initial cell as our root node, we set the `maximum_depth` of the game tree to be 4, guaranteeing each
player to move at least once, since a new game state will be created after each movement.
In other words, each layer of the game tree represents the movements of one player.
Moreover, we also looped through all possible numbers of the sheep being split to score highest.
Finally, we selected the action which would lead to the highest score at depth 4.

#### Result

The result, however, is much worse than expected, as the following game-play record suggests:

| iteration | player 19 | player 2 | player 3 | player 4 | winner |
|:---------:|-----------|----------|----------|----------|:------:|
|     1     | 24.03276  | 13.38604 | 25.0652  | 15.37319 |   3    |
|     2     | 24.33452  | 14.38604 | 8.47674  | 15.45434 |   19   |
|     3     | 24.03276  | 14.42497 | 18.14286 | 11.85516 |   19   |
|     4     | 14.1336   | 16.83276 | 8.94822  | 10.32664 |   2    |
|     5     | 18.45434  | 9        | 18.93171 | 17.69212 |   3    |
|     6     | 16.42497  | 17.27486 | 10.39051 | 19.78279 |   4    |
|     7     | 21.53668  | 13.38604 | 23.91509 | 11.70505 |   3    |
|     8     | 23.35079  | 9.39051  | 8.47674  | 16.42497 |   19   |
|     9     | 19.78098  | 13.94822 | 14.60508 | 12.89644 |   19   |
|    10     | 15.60508  | 10.03527 | 15.42497 | 14.08346 |   19   |

We can only secure `5` wins in `10` games, with an average score of `20.6865`, and win rate about `50%`.

## Discussion

### Enlarging The Reward Range Between The First Place and The Rest

The experiment data indicates improvement in model performance when the reward for the first place is notably larger
than the rewards for other ranks.
I believe the possible reason is that when there isn't a clear distinction between the rewards for other ranks and the
reward for the first place, insufficient simulation counts may lead to frequent visits to nodes with higher average
ranks but difficulty in winning the game.
This could result in the model making inferior choices.

### `Minimax` Method

`Minimax` may struggle to search through the entire game tree.
This becomes particularly evident when the game tree is large.
We can observe from the experimental data that when the search is highly incomplete, the performance of `Minimax` tends
to be unsatisfactory.
In contrast, `MCTS` can perform random simulations and selections based on a given time limit, making it more suitable
for making decisions within a limited time.

### Different Game Settings

#### Game 2

Due to the increase in `BOARD_SIZE` and `SHEEP_NUM` in game 2, the number of iterations that can be performed within the
same time limit decreases.
Consequently, this leads to a decrease in model performance.
In practical testing, since the size of the tree for the first step is the largest, it's noticeable that the model often
makes inferior choices when selecting the starting point.

#### Game 3

Since we cannot know the opponent's `sheepStat`, we must choose a reasonable way to ensure the model operates correctly.
Initially, we attempted to treat the number of sheep in each flock as infinite, meaning each simulation would continue
until the board was filled.
However, this approach led to longer simulation times, which led to fewer iterations and poor performance, and did not
properly simulate the actual gaming situation.
Therefore, we ultimately chose to evenly distribute the sheep among each flock, which resulted in better performance.

#### Game 4

As it was a team competition, we tried two approaches.
The first one involved averaging the sum of original rankings' rewards to determine the team's final reward.
The second approach assigned rewards based on the team's ranking.
Ultimately, the model's performance did not differ significantly between the two methods.
Therefore, we choose the second approach to implement.

### Potential Issues with Our Implementation

We noticed a potential issue with the model's decision-making during the final steps of the game.
It often makes poor choices near the end.
We suspect this might be due to how we designed our `MCTS`.
In our implementation, when expanding each layer of the tree, the child nodes represent all possible actions for that
player.
Consequently, if a player has no legal moves, the tree gets stuck at that layer without expanding nodes for other
players.
This limitation leads to poor performance by the model as the game approaches its conclusion.

A possible solution would be to skip the current player and proceed to expand nodes for the next player if the current
player has no available actions.
By doing so, we can prevent the tree from getting stuck and ensure that nodes are expanded for all players, thus
addressing the issue effectively.

## Conclusion

In this assignment, we learned various methods for implementing game agents, such as `Minimax`
and `Monte Carlo Tree Search (MCTS)`.
Throughout the implementation process, we gained a clearer understanding of the considerations and details required when
designing these trees.

## Contribution

- `110550032` 文玠敦
    1. Main logic implementation of `MCTS`, responsible for game 1 to 3.
    2. Producing and drafting of this report's `Discussion`, `Methodology`, `Conclusion`, `Implementation` from game 1
       to 3, and `Extreme Value for First Place` of `Experiment`.
    3. Propose and ran the `Extreme Value for First Place` of `Experiment`.
- `110550038` 莊書杰
    1. Main logic implementation of `Minimax`.
    2. Producing and drafting of this report's `Minimax Method` of `Experiment`.
    3. Propose and ran the `Minimax Method` of `Experiment`.
- `110550074` 馬楷翔
    1. Main logic adjustments to the implementation of `MCTS` game 4.
    2. Producing and drafting of this report's `Introduction`, `Original Setting` of `Experiment`, and `Contribution`.
    3. Ran the `Original Setting` of `Experiment`.

## Reference

You can find the source code and report on our [GitHub repository](https://github.com/seanmamasde/AICapstoneProject2).

<!--  
The report (maximum 10 pages single-spaced) should describe how your game agents work, your experiments and experiences,
similarities and differences in the different game settings, and contributions of individual team members. 
-->
