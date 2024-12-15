# Social Decision-Making and Helping Behavior: Can Q-Learning Models Learn to Help?

This repository contains the implementation and experiments for the project **"Social Decision-Making and Helping Behavior: Can Q-Learning Models Learn to Help?"**. 
The study builds upon the findings of Gureckis and Osborn Popp’s paper, exploring how Q-Learning and Deep Q-Learning can model altruistic behavior in a two-player game environment called the *Farm Task*.

## Overview

Helping behavior is central to human cooperation and social interactions. This project models helping gestures as sequential decision-making problems using reinforcement learning techniques. 
Specifically, it aims to:
- 1. Simulate helping behaviors using **Simple Tabular Q-Learning**, **Advanced Tabular Q-Learning**, and **Deep Q-Learning**.
- 2. Understand the influence of factors like **reciprocity**, **cost**, **resource capacity**, and **visibility** on decision-making.
- 3. Compare model predictions against empirical human behavior data.

---

## Methodology

### Simple Tabular Q-Learning
The **Simple Tabular Q-Learning** approach is a foundational implementation where a single Q-table is used to represent the state-action values for both agents. Key features include:
- **Training Setup**: 
  - Each episode represents a single player across all 12 randomized game environments.
  - Environmental factors like resource distribution, energy costs, and visibility conditions are randomized for each episode.
- **Reward Function**: The reward is computed as the sum of state-level rewards across all steps in an episode:
  \[
  \text{reward} = \sum_{i=1}^{n} \text{state.reward(current player)}_i
  \]
  where \(n\) is the number of steps.
- **Policy**: Uses an \(\epsilon\)-greedy policy to balance exploration and exploitation.
- **Limitations**: This approach models simplified helping behaviors but lacks the granularity to capture interactions between agents effectively.

Run Simple Tabular Q-Learning:
```
python train_simple_tabular_q.py
```

### Advanced Tabular Q-Learning
The Advanced Tabular Q-Learning approach improves upon the simple version with richer state representations and separate Q-tables for each agent. Key details:
- Two Q-Tables:
	•	Each agent maintains its Q-table, enabling personalized decision-making.

 - Reward Function:
[
r = \text{farm.reward(current player)} + \text{score} - (\text{distance discount} \times \text{step cost})
]
	•	Factors in the distance to vegetables and the energy cost for movement.
	•	step cost varies based on energy usage (low: 1, high: 2).

- State Representation:
	•	Includes energy levels, scores, backpack capacities, visibility, and helping status (redHelped, purpleHelped).

- Training:
	•	1,000 episodes with 12 randomized games per episode.
	•	Different resource, cost, and visibility conditions simulated.

- Advantages:
	•	Captures interactions between agents and reflects more realistic decision-making.

Run Advanced Tabular Q-Learning:

```
python train_advanced_tabular_q.py
```
