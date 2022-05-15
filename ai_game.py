from telnetlib import GA
import random
from collections import deque
from snakeai import SnakeGameAI
from SnakeGame.boards import GameBoard, DummyBoard
from model import Linear_QNet, QTrainer
import pygame
import click

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def load(self, file_name='./model/model.pth'):
        self.model = self.model.load(file_name)

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
            final_move = self.model.get_action(state)

        return final_move


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--model", "-m", type=str, help="Stored model File")
@click.option("--speed", "-s", type=int, help="pygame speed")
@click.option('--width', '-w', type=int, help='board width')
@click.option('--height', '-h', type=int, help='board height')
@click.argument("cmd", default='hide', nargs=1)
def train(**kwargs):
    """\n\t\t\tWecome to SnakegameAI\n
    * Click on the close control of the App, or hit Escpe to end the App\n
    """
    global agent, record
    speed = 500
    record = 0
    agent = Agent()
    speed = kwargs['speed'] or speed
    width = kwargs['width'] or 32
    height = kwargs['height'] or 24
    model_file = kwargs['model'] or None
    if kwargs['cmd'] == 'show':
        board = GameBoard(x=width, y=height)
        game = SnakeGameAI(board, speed)
    else:
        board = DummyBoard(x=width, y=height)
        game = SnakeGameAI(board)

    if model_file:
        agent.load(model_file)
        record = agent.model.record if hasattr(agent.model, 'record') else 0
        agent.n_games = agent.model.n_games if hasattr(agent.model, 'n_games') else 0

    while True:
        # get old state
        state_old = game.getState()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        alive, score, reward = game.moveTo(final_move)
        done = False if alive else True
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
