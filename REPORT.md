[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/15965062/44311825-11761100-a3f7-11e8-8412-5d14ee230bf7.png "Algorithm"
[image2]: https://raw.githubusercontent.com/lutaodai/DRL-Banana-Navigation/master/score.png "Plot of Rewards"

# Report - Deep RL Project: Navigation

### Implementation Details
Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### Setup of repository
Apart from the `README.md` file this repository consists of the following files:

1. `Navigation.ipynb`: An ipynb file for training the DQN agent and visualizing the training progress;
1. `model.py`: QNetwork class defining a DQN model;
1. `ddpg_agent.py`: Agent and ReplayBuffer classes; The Agent class makes use of the DQN model from `model.py` and the ReplayBuffer class;
1. `checkpoint.pth`: Contains the weights of the successful DQN model.


### Learning Algorithm

The agent is trained using the double-DQN algorithm.

References:
1. [DQN Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

2. [Double-DQN Paper](https://arxiv.org/abs/1509.06461)

3. Algorithm details: 

![Algorithm][image1]


4. Short explanation (refer to the papers for further details):
    - DQN: Adding experience replay (to decorrelate the experiences) and fixed target network (using second network that is updated slower then the main neural network) to the original Q learning algorithm.

    - Double-DQN: Using a different network to choose the argmax action from above algorithm (and since we have the fixed target network, we just use that). This is used to somewhat fix the problem of overestimation of Q values in the original DQN algoritm.
    
5. Hyperparameters used:
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
    - Aside from above, the code implemented here is almost ready for this challange, will only need to change the state space and load the AWS environment with X server (the provided env for Linux depends on X server).

2. Rainbow:
    - Several improvements to the DQN algorithm (in addition to Double-DQN) have risen over the years.
    - The [Rainbow paper](https://arxiv.org/abs/1710.02298) combines these ideas, adding some or all of them to this repo would be nice. 
