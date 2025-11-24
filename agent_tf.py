from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random
import os
import json

MAX_MEMORY = 100_000
BATCH_SIZE = 100
LR = 0.001

class Agent:
    def __init__(self, input_size=14, output_size=3, lr=LR, gamma=0.9):
        self.input_size = input_size
        self.lr = lr
        self.gamma = gamma # discount rate
        self.n_games = 0
        self.record = 0
        self.epsilon = 0 # randomness
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Sequential([
            Input(shape=(input_size, )),
            Dense(256, activation='relu'),
            Dense(output_size, activation='linear')
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.verbose = False
        self.model.compile(self.optimizer, self.loss)

    def save(self, dir_name='model'):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = os.path.join(dir_name, 'model.keras')
        self.model.save(file_name)
        # Store record and n_games on the Agent class, not the model
        info_file = file_name.replace('.keras', '.json')
        info_json = {'record': getattr(self, 'record', 0), 'n_games': self.n_games}
        print(f'saving the model for {self.n_games} games with record {getattr(self, "record", 0)}')
        with open(info_file, 'w') as f:
            f.write(json.dumps(info_json))
    
    def load(self, dir_name='model'):
        print('loading the stored model')
        file_name = os.path.join(dir_name, 'model.keras')
        # Check if file exists before loading
        if not os.path.exists(file_name):
            print(f"Warning: Model file {file_name} not found")
            return
            
        # Load weights with the correct format
        self.model = load_model(file_name)
        
        # Load metadata
        info_file = file_name.replace('.keras', '.json')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info_json = json.load(f)
                # Set attributes on the Agent class, not the model
                if 'record' in info_json:
                    self.record = info_json['record']
                if 'n_games' in info_json:
                    self.n_games = info_json['n_games']
        else:
            print(f"Warning: Info file {info_file} not found")

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

    def train_step(self, state, action, reward, next_state, alive):
        # 1: Model's prediction outputs Q-values for all possible actions.
        #    Output layer uses LINEAR activation (not softmax like classification tasks)
        #    target = [Q₀, Q₁, Q₂], one Q-value per action (unbounded real numbers)
        #    During inference, the action with highest Q-value is taken
        target = np.array(self.model(np.array(state)))
        
        self.verbose and print('Q Learning')
        self.verbose and print('\tbefore : ', target)          # Q-values before incorporating actual experience
        
        for idx in range(len(alive)):
            # 2: Calculate TD target using Bellman equation: Q_new = r + γ * max(Q(next_state))
            #    Only include future reward if episode continues (alive[idx] == True)
            delayed_reward = self.gamma * np.amax(self.model(np.array([next_state[idx]]))) if alive[idx] else 0
            Q_new = reward[idx] + delayed_reward
            
            # 3: Replace only the Q-value of the action that was actually taken
            #    Other Q-values remain as model's current predictions
            target[idx][np.argmax(action[idx]).item()] = Q_new
        
        self.verbose and print('\tafter : ', target)          # Q-values after incorporating observed reward
        
        # Calculate gradient: minimize difference between predicted Q and TD target
        self.model.fit(np.array(state), np.array(target), verbose=False)
