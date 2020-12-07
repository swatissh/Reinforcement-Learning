import argparse
import random
from environment import TreasureCube
import numpy as np
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt

class GreedyPolicy():
    '''
    Creates a epsilon(exploration rate)-greedy policy. 
    The probability function takes the exploration rate, length 
    of action_space, state and q-table as input and returns 
    the probabilities for each action in the given state.

    '''   
    
    def probability(self, exploration_rate, len_action_space, q_table, agent_state):
        self.explorationRate = exploration_rate
        self.lenActionSpace = len_action_space
        action_prob = np.ones(self.lenActionSpace, dtype=float) * self.explorationRate / self.lenActionSpace
        best_action = np.argmax(q_table[agent_state])
        action_prob[best_action] += (1 - self.explorationRate)
        return action_prob

# you need to implement your agent based on one RL algorithm
class RandomAgent(object):
    def __init__(self):
        self.action_space = ['left','right','forward','backward','up','down'] # in TreasureCube
        self.discount_factor = 0.99
        self.learning_rate = 0.5
        self.exploration_rate = 0.01
        self.Q =  defaultdict(lambda : np.zeros(len(self.action_space))) #empty Q initially 
        self.policy = GreedyPolicy()    #GreedyPolicy

    def take_action(self, state):
        action_probability = self.policy.probability(self.exploration_rate, len(self.action_space), self.Q, state) 
        #action is chosen according to the probability distribution
        index = np.random.choice(np.arange(len(action_probability)), p=action_probability) #index is int within 0 to 5
        if(index==0):
            action = 'left'
        elif(index==1):
            action = 'right'
        elif(index==2):
            action = 'forward'
        elif(index==3):
            action = 'backward'
        elif(index==4):
            action = 'up'
        else:
            action = 'down'

        return action

    # Convert action_space string to index and pass index to train( )
    def getActionIndex(self,action):
        self.action = action
        index = 0
        if(self.action=='left'):
            index = 0
        elif(self.action=='right'):
            index = 1
        elif(self.action=='forward'):
            index = 2
        elif(self.action=='backward'):
            index = 3
        elif(self.action=='up'):
            index = 4
        else:
            index = 5
        
        return index

    # implement your train/update function to update self.V or self.Q
    # you should pass arguments to the train function
    def train(self, state, action, next_state, reward):
        next_action = np.argmax(self.Q[next_state])
        self.Q[state][action] += self.learning_rate * ((reward + self.discount_factor * self.Q[next_state][next_action]) - self.Q[state][action])

    
def test_cube(max_episode, max_step):
    env = TreasureCube(max_step=max_step)
    agent = RandomAgent()

    episodeReward = [] #Keep track of episode reward from each episode for plotting graph
    stepCount = []     # Keep track of total_steps from each episode for plotting graph
    countEpisode = []  # Keep track of number of episodes for plotting graph

    for epsisode_num in range(0, max_episode):
        state = env.reset()
        terminate = False
        t = 0
        episode_reward = 0
        while not terminate:
            action = agent.take_action(state)
            reward, terminate, next_state = env.step(action)
            episode_reward += reward
            # you can comment the following two lines, if the output is too much
            #env.render() # comment
            #print(f'step: {t}, action: {action}, reward: {reward}') # comment
            t += 1
            action_index = agent.getActionIndex(action) #convert action_spaces string to index
            agent.train(state, action_index, next_state, reward)
            state = next_state
        print(f'epsisode: {epsisode_num}, total_steps: {t}, episode reward: {episode_reward}')
        episodeReward.append(episode_reward)
        stepCount.append(t)
        countEpisode.append(epsisode_num)

    print("--------------------------------------Q-Table--------------------------------------------")  
    q_table_dict = dict(agent.Q)
    for s, q in q_table_dict.items():
        print('State: {} Q-value: {}'.format(s, q))
    print("-----------------------------------------------------------------------------------------")

    print("----------------------------------Learning Progress Info---------------------------------")  
    print("Average epsiode reward:", statistics.mean(episodeReward))
    print("Average step count:", statistics.mean(stepCount))
    print("-----------------------------------------------------------------------------------------")

    plotLearningProgess(episodeReward,countEpisode) #Plot learning progress graph


# Plot Learning Progress graph
def plotLearningProgess(episodeReward, countEpisode):
    y = episodeReward
    x = countEpisode
    plt.title("Learning Progress: Episode rewards vs. Episodes")  
    plt.xlabel("Episodes")  
    plt.ylabel("Episode rewards")  
    plt.plot(x, y, color ="deepskyblue")  
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--max_episode', type=int, default=500)
    parser.add_argument('--max_step', type=int, default=500)
    args = parser.parse_args()

    test_cube(args.max_episode, args.max_step)
