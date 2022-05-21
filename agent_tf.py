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

class Linear_QNet:
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

class Agent:
    def __init__(self, input_size=14, output_size=3, lr=LR, gamma=0.9):
        self.lr = lr
        self.gamma = gamma # discount rate
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Sequential([
            Flatten(input_shape=(1,input_size)),
            Dense(256, activation='relu'),
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
        self.train_step(states, actions, rewards, next_states, alive)

    def trainShortMemory(self, state, action, reward, next_state, alive):
        self.train_step(state, action, reward, next_state, alive)

    def getAction(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            final_move = [0,0,0]
            #prediction = self.model.predict(state.reshape(1,14))
            #move = np.argmax(prediction)
            move = random.randint(0, 2)
            final_move[move] = 1
            return final_move

        return final_move

    def train_step(self, state, action, reward, next_state, alive):
        if len(state.shape) == 1:
            # this is for the short memory. ie. each move
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            alive = (alive, )

        # 1: predicted Q values with current state
        print(state.shape)
        pred = self.model.predict(state.reshape((1,14)))

        target = pred.clone()
        for idx in range(len(alive)):
            Q_new = reward[idx]
            if alive[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()