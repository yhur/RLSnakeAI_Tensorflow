import pygame
import sys
from snakeai import SnakeGameAI
from SnakeGame.boards import GameBoard, Board
from agent_tf import Agent
import click
import signal
import os, shutil

def handler(signum, frame):
    sys.exit()
signal.signal(signal.SIGINT, handler)

@click.command(context_settings=dict(help_option_names=["-h", "--help"]), help="\nEx)python game_tf.py -m model/model.weights.h5 -s 500 -w 32 -h 24 show\n")
@click.option("--model", "-m", type=str, help="Stored model File")
@click.option("--speed", "-s", type=int, help="pygame speed")
@click.option('--width', '-w', type=int, help='board width')
@click.option('--board_height', '-b', type=int, help='board height')
@click.option('--verbose', '-v', is_flag=True, help="Enable verbose mode.")
@click.argument("cmd", default='hide', nargs=1)
def train(**kwargs):
    """\n\t\t\tWecome to SnakegameAI\n
    * Click on the close control of the App, or hit Escpe to end the App\n
    """
    #global agent, record
    speed = 500
    speed = kwargs['speed'] or speed
    width = kwargs['width'] or 32
    height = kwargs['board_height'] or 24
    model_dir = kwargs['model'] or None
    agent = Agent()
    agent.verbose = kwargs['verbose']
    #agent.verbose = True
    if kwargs['cmd'] == 'show':
        board = GameBoard(x=width, y=height, speed=speed)
        game = SnakeGameAI(board)
    else:
        board = Board(x=width, y=height)
        game = SnakeGameAI(board)

    if model_dir:
        if os.path.exists(model_dir):
            agent.load(model_dir)
            print(f"\tModel '{model_dir}' loaded(n_game:{agent.n_games}, record score:{agent.record})")
        else:
            print(f"\n\n\tModel '{model_dir}' doesn't exist\n\n")
            if input("Is this a fresh start? y/n") != 'y':
                sys.exit()
    else:
        model_dir = 'model'
        if os.path.exists(model_dir):
            if input(f"\tModel '{model_dir}' exists. Do you want to delete and restart? y/n") == 'y':
                shutil.rmtree(model_dir)
            else:
                agent.load(model_dir)
                print(f"\tModel '{model_dir}' loaded(n_game:{agent.n_games}, record score:{agent.record})")

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

            if score > agent.record:
                agent.record = score
                agent.save(model_dir)

            agent.verbose and print('Game', agent.n_games, 'Score', score, 'Record:', agent.record)

if __name__ == '__main__':
    train()