## Grid World Gym

simple multi-agent gridworld gym loaded from text file maps with simple collision detection and prevention. This is the environment used for the gridworld environment in [Safe multi-agent reinforcement learning via shielding](https://arxiv.org/pdf/2101.11196).

#### Relevant repositories:
- [Shielded MARL](https://github.com/IngyN/Shield_MARL)
- [Shielded Deep MARL](https://github.com/IngyN/Shielded_DMARL)

#### Notes:

- Step function:
  - The reward for a collision can be set where `collision_cost` = -reward. If you want the reward of a collision to be -10 then `collision_cost=10`. 
  - Penalty for unmoving agents can be set by having `noop=True`
  - The `share` variable should be `True` for shared goals (ie. if multiple agents are going to one goal location).
  - Out of bounds checking with penalty -10.
  - Random priority of agents for conflicting agent actions can be enabled with `random_priority=True`.


Requires the OpenAI gym package. 
