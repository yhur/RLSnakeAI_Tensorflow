import numpy as np
from SnakeGame.snake import Snake, Direction, Point

class SnakeGameAI(Snake):
    def __init__(self, board):
        super().__init__(board)

    def moveTo(self, action):
        reward = 0
        old_score = self.score
        gameOn = super().moveTo(action)
        if gameOn == False:
            reward -= 10
        elif self.score > old_score:
            reward += 10
        return gameOn, self.score, reward

    def rightDirection(self, direction=None):
        direction = direction if direction else self.direction
        idx = self.directionRing.index(direction)
        return self.directionRing[(idx + 1) % 4]

    def leftDirection(self, direction=None):
        direction = direction if direction else self.direction
        idx = self.directionRing.index(direction)
        return self.directionRing[(idx - 1) % 4]

    def getDistanceToWall(self, direction):
        """Returns normalized distance to wall in given direction"""
        current = Point(self.head.x, self.head.y)
        distance = 0
        max_distance = max(self.x, self.y)
        
        while True:
            current = Point(current.x + direction.value[0], current.y + direction.value[1])
            distance += 1
            
            # Hit wall
            if current.x >= self.x or current.x < 0 or current.y >= self.y or current.y < 0:
                return distance / max_distance

    def getDistanceToBody(self, direction):
        """Returns normalized distance to body in given direction (1.0 if no body found)"""
        current = Point(self.head.x, self.head.y)
        distance = 0
        max_distance = max(self.x, self.y)
        
        while True:
            current = Point(current.x + direction.value[0], current.y + direction.value[1])
            distance += 1
            
            # Hit wall
            if current.x >= self.x or current.x < 0 or current.y >= self.y or current.y < 0:
                return 1.0
            
            # Hit body
            if current in self.body[1:]:
                return distance / max_distance

    def getState(self):
        forward = self.direction
        right = self.rightDirection()
        left  = self.leftDirection()
        
        state = [
            # Wall distances (normalized)
            self.getDistanceToWall(forward),
            self.getDistanceToWall(right),
            self.getDistanceToWall(left),
            # current moving direction as one hot encoding [R, D, L, U]
            *[self.direction == d for d in list(Direction)],
            # Body distances (normalized, 1.0 if no body in that direction)
            self.getDistanceToBody(forward),
            self.getDistanceToBody(right),
            self.getDistanceToBody(left),
            # Food location 
            self.board.apple.x < self.head.x,  # food left
            self.board.apple.x > self.head.x,  # food right
            self.board.apple.y < self.head.y,  # food up
            self.board.apple.y > self.head.y   # food down
        ]
        return np.array(state, dtype=float)