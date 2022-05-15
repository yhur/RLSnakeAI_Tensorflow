import torch
import random
from collections import deque
from game import SnakeGameAI
from model import Linear_QNet, QTrainer
from helper import plot
import pygame
import click

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

clock = pygame.time.Clock()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def load(self, file_name='./model/model.pth'):
        print('loading the stored weights')
        self.model = torch.load(file_name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--weights", "-w", type=str, help="Weights File")
@click.option("--speed", "-s", type=int, help="pygame speed")
@click.option('--bsize', '-b', type=(int, int), help='board size')
def train(**kwargs):
    """\n\t\t\tWecome to SnakegameAI\n
    * Click on the close control of the App, or hit Escpe to end the App\n
    """
    global agent, record
    speed = 500
    record = 0
    agent = Agent()
    speed = kwargs['speed'] or speed
    bsize = kwargs['bsize'] or (32, 24)
    weights = kwargs['weights'] or None
    game = SnakeGameAI(x=bsize[0], y=bsize[1])
    if weights:
        agent.load(weights)
        record = agent.model.record if hasattr(agent.model, 'record') else 0
        agent.n_games = agent.model.n_games if hasattr(agent.model, 'n_games') else 0

    while True:
        # get old state
        state_old = game.getState()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.moveTo(final_move)
        clock.tick(speed)
        state_new = game.getState()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.n_games = agent.n_games
                agent.model.record = record
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    try:
        train()
    except pygame.error as e:
        agent.model.n_games = agent.n_games
        agent.model.record = record
        agent.model.save()
        print('App stopped')
