import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque, namedtuple
import torch
import argparse
import os

from unityagents import UnityEnvironment
from dqn_agent import Agent, ReplayBuffer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-dou_dqn', '--double_dqn', default=False, type=str2bool, 
                    help="specifying if using double dqn")
PARSER.add_argument('-due_dqn', '--dueling_dqn', default=False, type=str2bool, 
                    help="specifying if using dueling dqn")
PARSER.add_argument('-fdir', '--figure_dir', default="figure",
                    help="directory storing figures")
PARSER.add_argument('-mdir', '--model_dir', default='model',
                    help="directory storing models")
PARSER.add_argument('-ws', '--window_size', default=100, type=int,
                    help="moving average window for plotting")
PARSER.add_argument('-bus', '--buffer_size', default=int(1e5), type=int,
                    help='buffer size for experience replay buffer')
PARSER.add_argument('-bas', '--batch_size', default=256, type=int,
                    help='batch size training')
PARSER.add_argument('-gamma', '--gamma', default=0.99, type=float,
                    help='discount factor')
PARSER.add_argument('-tau', '--tau', default=1e-3, type=float,
                    help='factor for soft update of target parameters')
PARSER.add_argument('-lr', '--lr', default=5e-4, type=float,
                    help='learning rate')
PARSER.add_argument('-uf', '--update_frequency', default=4, type=int,
                    help='how often to update the network')
ARGS = PARSER.parse_args()
print(ARGS)

if not os.path.exists(ARGS.figure_dir): os.makedirs(ARGS.figure_dir)
if not os.path.exists(ARGS.model_dir): os.makedirs(ARGS.model_dir)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0] 
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            
            if ARGS.double_dqn:
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(ARGS.model_dir, 'model_double_dqn.pth'))
            elif ARGS.dueling_dqn:
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(ARGS.model_dir, 'model_dueling_dqn.pth'))
            else:
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(ARGS.model_dir, 'model_dqn.pth'))
            break
    return scores

def plot_scores(scores, window_size=15):
    scores = pd.DataFrame(scores, columns=["scores"])
    scores = scores.reset_index()
    scores["scores_avg"] = scores["scores"].rolling(window=window_size).mean()
    
    sns.set_style("dark")
    sns.relplot(x = "index", y = "scores",
                data=scores, kind="line")
    plt.plot(scores["index"], scores["scores_avg"], color=sns.xkcd_rgb["amber"])
    plt.legend(["Scores", "MA(%d)" %window_size])
    
    if ARGS.double_dqn:
        plt.savefig(os.path.join(ARGS.figure_dir, "score_plot_double_dqn.png"))
    elif ARGS.dueling_dqn:
        plt.savefig(os.path.join(ARGS.figure_dir, "score_plot_dueling_dqn.png"))
    else:
        plt.savefig(os.path.join(ARGS.figure_dir, "score_plot_dqn.png"))
    

if __name__ == "__main__":    
    env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64",
                       worker_id=1, seed=1)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # initialize an agent
    agent = Agent(state_size=37, action_size=4, seed=1, args=ARGS)
    
    # training a dqn agent
    scores = dqn(n_episodes=3000, max_t=1000)
    
    # visualization
    plot_scores(scores, window_size=ARGS.window_size)