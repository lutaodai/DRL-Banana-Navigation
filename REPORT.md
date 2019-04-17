[//]: # (Image References)

[image1]: https://cdn-images-1.medium.com/max/1600/1*nb61CxDTTAWR1EJnbCl1cA.png "Algorithm"
[image2]: https://raw.githubusercontent.com/lutaodai/DRL-Banana-Navigation/master/score.png "Plot of Rewards"


# Report - Deep RL Project: Navigation


### Implementation Details
Implementation details, including results and the score plot can be found in the `Navigation.ipynb`.  

### Setup of repository
Apart from the `README.md` file this repository consists of the following files:

1. `Navigation.ipynb`: An ipynb file for training the DQN agent and visualizing the training progress;
1. `model.py`: QNetwork class defining a DQN model;
1. `ddpg_agent.py`: Agent and ReplayBuffer classes; The Agent class makes use of the DQN model from `model.py` and the ReplayBuffer class;
1. `checkpoint.pth`: Contains the weights of the successful DQN model.


### Learning Algorithm: Deep Q-Networks

The agent is trained using the [Deep Q-Networks](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

The Deep Q-Learning algorithm represents the optimal action-value function by a neural network, instead of a table. However, reinforcement learning is notoriously unstable when neural networks are used to represent the action values. To stablize the learning, Deep Q-Learning algorithm introduces two structures
    - experience replay: an allocated fixed-size memeory to store past experience. A random experience sample is drawn each time to break correlation and increase data efficiency
    - fixed Q-target: an model with identical structure but being updated much slowly or over several episodes. The target network is used to calcualte the TD target.

1. Algorithm details: 

![Algorithm][image1]

    
2. Hyperparameters used:
    ```
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 256        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    ```
### Screen outputs
```
Episode 100	Average Score: 0.75
Episode 200	Average Score: 3.66
Episode 300	Average Score: 6.27
Episode 400	Average Score: 9.72
Episode 500	Average Score: 11.87
Episode 557	Average Score: 13.02
Environment solved in 457 episodes!	Average Score: 13.02
```

### Plot of results

As seen below, the agent solves the environment after 457 episodes, and achieves best average score of above 13.

![Plot of Rewards][image2]

###  Ideas for future work

1. Learning from Pixels:
    - Learning from pixels instead of the given states will require a different network architecture (CNN based) and additional training time.

2. Improvement to DQN algorithm:
    - Implementing [prioritized experience replay](https://arxiv.org/abs/1511.05952) instead of random sampling to achieve a better and quicker convergence;
    - Implementing [double-DQN](https://arxiv.org/abs/1509.06461) to address the issue of Q value overestimation;
    - Implementing [dueling-DQN](https://arxiv.org/abs/1511.06581) to estimate the state value function and the advantage function instead of the Quality function. It would be interesting to see if it would lead to a more stable learning.

3. Tuning the model and hyperparameters, including
    - Modifying Model depth and width. Currently the neural network is still relatively shallow and narrow (3 layers with each layer having 64 neurons or below). It would be interesting to see whether a more flexible model would lead to quicker convergence or on the contrary, unstability)
    - Adding normalization layers, such as batch normalization layers
    - Tuning the hyperparameters, such as gamma, tau, learning rate.
