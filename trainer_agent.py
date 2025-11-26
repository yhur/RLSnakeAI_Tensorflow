import os
import json
import numpy as np
from agent_tf import Agent
import random

BATCH_SIZE = 1000

class TrainerAgent(Agent):
    def remember(self, state, action, reward, next_state, alive, trainee_state):
        self.memory.append((state, action, reward, next_state, alive, trainee_state)) # popleft if MAX_MEMORY is reached


    def trainShortMemory(self, state, action, reward, next_state, alive, trainee_state):
        state, action, reward, next_state, alive, trainee_state = [state], [action], [reward], [next_state], [alive], [trainee_state]
        self.train_step(state, action, reward, next_state, alive, trainee_state)

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, alives, trainee_state = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, alives, trainee_state)

    def train_step(self, state, action, reward, next_state, alive, trainee_state):
        target = np.array(self.model(np.array(state)))
        
        self.verbose and print('Q Learning')
        self.verbose and print('\tbefore : ', target)
        
        for idx in range(len(alive)):
            delayed_reward = self.gamma * np.amax(self.model(np.array([next_state[idx]]))) if alive[idx] else 0
            Q_new = reward[idx] + delayed_reward
            target[idx][np.argmax(action[idx]).item()] = Q_new
        
        self.verbose and print('\tafter : ', target)
        
        self.trainee.fit(np.array(trainee_state), np.array(target), verbose=False)