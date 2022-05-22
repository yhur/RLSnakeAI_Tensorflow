import pygame
import sys
from snakeai import SnakeGameAI
from SnakeGame.boards import GameBoard, Board
from agent_torch import Agent
import click
import signal
import os

def handler(signum, frame):
    sys.exit()
signal.signal(signal.SIGINT, handler)

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
    #global agent, record
    speed = 500
    record = 0
    agent = Agent()
    speed = kwargs['speed'] or speed
    width = kwargs['width'] or 32
    height = kwargs['height'] or 24
    model_file = kwargs['model'] or None
    if kwargs['cmd'] == 'show':
        board = GameBoard(x=width, y=height, speed=speed)
        game = SnakeGameAI(board)
    else:
        board = Board(x=width, y=height)
        game = SnakeGameAI(board)

    if model_file:
        if os.path.exists(model_file):
            agent.load(model_file)
            record = agent.model.record if hasattr(agent.model, 'record') else 0
            agent.n_games = agent.model.n_games if hasattr(agent.model, 'n_games') else 0
        else:
            print(f"\n\n\tModel file {model_file} doesn't exist\n\n")
            sys.exit()

    while True:
        # keyboard handling to capture the ending of the App
        if pygame.display.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

        # get current state
        state0 = game.getState()

        # get action
        action = agent.getAction(state0)

        # perform move and get new state
        alive, score, reward = game.moveTo(action)
        state1 = game.getState()

        # train short memory
        agent.trainShortMemory(state0, action, reward, state1, alive)

        # remember
        agent.remember(state0, action, reward, state1, alive)

        if alive == False:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.trainLongMemory()

            if score > record:
                record = score
                agent.model.n_games = agent.n_games
                agent.model.record = record
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    train()