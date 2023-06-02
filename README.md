# laikago_obot

## composition

The project has 4 modules, which are
- robot: simulation of mechanical and electrical characteristics
- transfer: Data processing from robot to Gym environment
- builder: encapsulation of Gym environment
- agent: learning of deep reinforcement learning algorithm

These four modules are described in detail below

###robot

In this module, we use a python program to simulate a laikago as accurately as possible where C++ is using in the real environment. The task of this module is to make the latter three modules (at the program level) in the real environment and the simulation environment runs without distinction.

In order to make our modeling as accurate as possible, this layer uses two methods so that the agent trained in the simulation environment can run in the real environment:
- system identification: perfect the urdf file of laikago as much as possible, and model the system **delay time**
- randomization: random perturbation of mechanical and electrical properties, as well as random forces applied from the outside (although there are data to prove that the two are equivalent)

At the same time, we use torque control instead of position control at this layer, because we need to improve the PD algorithm ourselves, and unitree has encapsulated the bottom layer so that we cannot modify it.
Calculating torque by position is one of the tasks of transfer.

The robot provides two observations, namely:
- get_true_observation: real observation, used to calculate reward
- get_observation: Observations with random noise added, used as input to the neural network

### transfer

After this module, the codes of the simulation environment and the real environment are exactly the same.

The accusations of this module mainly include:
- Calculate the torque to control the motor through the position returned by the upper module (Builder)
- Accumulate robot's historical data to help agent deal with POMDP problems
- Calculate more obs through the underlying observation, and provide an interface for reward calculation

### builder

Combining the underlying data from the transfer layer with the task, and meeting the interface specifications of the env class in the gym, forms an environment that can be directly used for reinforcement learning algorithm learning.
