import numpy as np
from SnakeGame.snake import Snake, Direction, Point

class SnakeGameAI(Snake):
    def moveTo(self, action):
        reward = 0
        old_score = self.score
        game_over, score = super().moveTo(action)
        if game_over:
            reward -= 10
        elif score > old_score:
            reward += 10
        return reward, game_over, score

    def rightDirection(self, direction=None):
        direction = direction if direction else self.direction
        idx = self.directionRing.index(direction)
        return self.directionRing[(idx + 1) % 4]

    def leftDirection(self, direction=None):
        direction = direction if direction else self.direction
        idx = self.directionRing.index(direction)
        return self.directionRing[(idx - 1) % 4]

    def getState(self):
        forward = self.direction
        right = self.rightDirection()
        left  = self.leftDirection()

        state = [
            # Danger straight
            self.is_coliding(Point(self.head.x + forward.value[0], self.head.y + forward.value[1])),
            # Danger right
            self.is_coliding(Point(self.head.x + right.value[0], self.head.y + right.value[1])),
            # Danger left
            self.is_coliding(Point(self.head.x + left.value[0], self.head.y + left.value[1])),
            # Move direction
            *[self.direction == d for d in list(Direction)],
            # Food location 
            self.apple.x < self.head.x,  # food left
            self.apple.x > self.head.x,  # food right
            self.apple.y < self.head.y,  # food up
            self.apple.y > self.head.y   # food down
        ]
        return np.array(state, dtype=int)