from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
import os
import json

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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        #self.model.compile(Adam(learning_rate=lr), loss='mse')

    def save(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        info_file = file_name.split('.h5')[0] + '.json'
        info_json = {'record': self.model.record, 'n_games': self.model.n_games}
        with open(info_file, 'w') as f:
            f.write(json.dumps(info_json))
        self.model.save_weights(file_name)

    def load(self, file_name='./model/model.h5'):
        print('loading the stored model')
        self.model.load_weights(file_name)
        info_file = file_name.split('.h5')[0] + '.json'
        with open(info_file, 'r') as f:
            info_json = json.load(f)
            if 'record' in info_json:
                self.model.record = info_json['record']
            if 'n_games' in info_json:
                self.model.n_games = info_json['n_games']

    def remember(self, state, action, reward, next_state, alive):
        self.memory.append((state, action, reward, next_state, alive)) # popleft if MAX_MEMORY is reached

    def trainLongMemory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, alives = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, alives)

    def trainShortMemory(self, state, action, reward, next_state, alive):
        state, action, reward, next_state, alive = [state], [action], [reward], [next_state], [alive]
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
            prediction = self.model(np.array([state]))
            move = np.argmax(prediction)
            final_move[move] = 1
            return final_move

        return final_move

    def train_step(self, state, action, reward, next_state, alive):
        # 1: predicted Q values with current state
        target = np.array(self.model(np.array(state)))

        print('Q Learning')
        print('\tbefore : ', target)          # target Q with only Immediate Reward
        for idx in range(len(alive)):
            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # preds[argmax(action)] = Q_new
            Q_new = reward[idx]
            if alive[idx]:
                Q_new = reward[idx] + self.gamma * np.amax(self.model(np.array([next_state[idx]])))

            target[idx][np.argmax(action[idx]).item()] = Q_new
        print('\tafter : ', target)          # target Q with the delayed Reward
    
        with tf.GradientTape() as tape:
            pred = self.model(np.array(state))
            main_loss = tf.reduce_mean(self.loss(target, pred))
            loss = tf.add_n([main_loss] + self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        #self.model.fit(np.array(state), np.array(target), verbose=False)