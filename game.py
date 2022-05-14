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

    def hitWall(self, direction):
        x = self.head.x + direction.value[0]
        y = self.head.y + direction.value[1]
        if x >= self.x or x < 0 or y >= self.y or y <0:
            return True
        else:
            return False

    def bodyCheck(self, direction):
        def bodyAhead(current, direction):
            newPoint = Point(current.x + direction.value[0], current.y + direction.value[1])
            #print(current, newPoint, direction)
            if newPoint.x >= self.x or newPoint.x < 0 or newPoint.y >= self.y or newPoint.y < 0:
                return False
            else:
                if newPoint in self.body[1:]:
                    return True
                else:
                    return bodyAhead(newPoint, direction)
        return bodyAhead(self.head, direction)
        #return Point(self.head.x + direction.value[0], self.head.y + direction.value[1]) in self.body[1:]

    def getState(self):
        forward = self.direction
        right = self.rightDirection()
        left  = self.leftDirection()

        state = [
            # Danger straight - Wall
            self.hitWall(forward),
            # Danger right - Wall
            self.hitWall(right),
            # Danger left - Wall
            self.hitWall(left),
            # Move direction
            *[self.direction == d for d in list(Direction)],
            # Danger straight - body
            self.bodyCheck(forward),
            self.bodyCheck(right),
            self.bodyCheck(left),
            # Food location 
            self.apple.x < self.head.x,  # food left
            self.apple.x > self.head.x,  # food right
            self.apple.y < self.head.y,  # food up
            self.apple.y > self.head.y   # food down
        ]
        return np.array(state, dtype=int)