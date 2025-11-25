# Main Requirements

This code base contains interfaces and implementation of PPO algorithms. Implementations
rely on the Burn Rust framework for model training and inference.

1. Environment trait should support agent configurations
   1. One agent interacts with the environment 
   1. Many agents interact with the environment 
   1. Agents can be opponents or can cooperate 
   1. Agents can interact on turn base approach or make simultaneous actions
   
1. Actions
   1. Actions can be discrete or continuous
   1. Each agent can have one or more actions per turn/step
   1. Agent actions can be continuous or discrete or both

1. Observations
   1. Observation should be per agent. 
   1. Each agent can have the same or different observation
   1. Observation input layer can have different lengths per agent model
   1. Action space can be different per agent
   
1. Neural Network Model
   1. Each agent can have a shared or individual NN(Neural Network) model
   1. Neural Network model should have a well-defined train / interface so that implementation of different algorithms is straightforward

1. Supported Reinforcement Learning algorithms 
   1. PPO
   2. SAC
   3. Other (abstraction should allow implementing different algorithms)