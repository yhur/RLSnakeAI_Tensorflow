from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
import os
import json
import numpy as np
from agent_tf import Agent

N_TRANSFER = 100

def add_weights(org_model, agent):
    def load(model, dir_name='model'):
        model_file = os.path.join(dir_name, 'model.keras')
        info_file  = os.path.join(dir_name, 'model.json')
        for f in (model_file, info_file):
            if not os.path.exists(f):
                print(f"Warning: Missing file: {f}")
                return

        model.set_weights(load_model(model_file).get_weights())
        
        with open(info_file, 'r') as f:
            info_json = json.load(f)
            record = info_json['record'] if 'record' in info_json else 0
            n_games = info_json['n_games'] if 'n_games' in info_json else 0
    
        return record, n_games

    o_model = Sequential([
        Input(shape=(14, )),
        Dense(256, activation='relu'),
        Dense(3, activation='linear')
    ])
    agent.record, agent.n_games = load(o_model, org_model)
    w=o_model.get_weights()
    w[0] = np.vstack([w[0], np.random.randn(3, 256)* np.sqrt(2/(17+256))])
    agent.model.set_weights(w)
    agent.model.compile(agent.optimizer, agent.loss) 

class TransferAgent(Agent):
    def freeze(self):
        self.frozen = self.model.get_weights()       # get the original weights to freeze

    def apply_frozen(self, cut=-1):
        w_new = self.model.get_weights()[0][cut]    # the 1st hidden layer's weights for the last input neurons 
        self.frozen[0] = np.delete(self.frozen[0], cut, axis=0)   # the last n lines of the weight 1 (cut)
        self.frozen[0] = np.vstack([self.frozen[0], w_new])
        self.model.set_weights(self.frozen)

    def train_step(self, state, action, reward, next_state, alive):
        target = np.array(self.model(np.array(state)))
        
        self.verbose and print('Q Learning')
        self.verbose and print('\tbefore : ', target)
        
        for idx in range(len(alive)):
            delayed_reward = self.gamma * np.amax(self.model(np.array([next_state[idx]]))) if alive[idx] else 0
            Q_new = reward[idx] + delayed_reward
            target[idx][np.argmax(action[idx]).item()] = Q_new
        
        self.verbose and print('\tafter : ', target)
        
        self.model.fit(np.array(state), np.array(target), verbose=False)
        # Apply the frozen to the existing weights
        self.apply_frozen()