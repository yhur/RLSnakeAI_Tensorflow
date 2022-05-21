from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, input_size=14, output_size=3, lr=LR, gamma=0.9):
        self.input_size = input_size
        self.lr = lr
        self.gamma = gamma # discount rate
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Sequential([
            Dense(256, input_shape=(input_size, ), activation='relu'),
            Dense(output_size, activation='linear')
        ])
        self.model.compile(Adam(learning_rate=lr), metrics=['mae'])

    def load(self, file_name='./model/model.pth'):
        pass

    def load(self, file_name='./model/model.pth'):
        pass

    def remember(self, state, action, reward, next_state, alive):
        self.memory.append((state, action, reward, next_state, alive)) # popleft if MAX_MEMORY is reached

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, alives = zip(*mini_sample)
        print("hhhhhh")
        self.train_step(states, actions, rewards, next_states, alives)

    def trainShortMemory(self, state, action, reward, next_state, alive):
        state, action, reward, next_state, alive = [state], [action], [reward], [next_state], [alive]
        self.train_step(state, action, reward, next_state, alive)

    def getAction(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        print('get 1')
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            print('get 2')
            final_move = [0,0,0]
            prediction = self.model.predict(np.array([state]))
            move = np.argmax(prediction)
            final_move[move] = 1
            return final_move

        return final_move

    def train_step(self, state, action, reward, next_state, alive):
        # 1: predicted Q values with current state
        pred = self.model.predict(np.array(state))

        print('pred len', len(pred))
        print('pred type', type(pred))

        target = pred.copy()
        for idx in range(len(alive)):
            Q_new = reward[idx]
            if alive[idx]:
                print('in q', next_state[idx])
                Q_new = reward[idx] + self.gamma * np.amax(self.model.predict(np.array([next_state[idx]])))

            target[idx][np.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        #self.model.fit(target, pred)
        self.model.fit(np.array(state), np.array(target))