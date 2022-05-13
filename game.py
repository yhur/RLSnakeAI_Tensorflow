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

    def getState(self):
        point_l = Point(self.head.x - 1, self.head.y)
        point_r = Point(self.head.x + 1, self.head.y)
        point_u = Point(self.head.x, self.head.y - 1)
        point_d = Point(self.head.x, self.head.y + 1)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_coliding(point_r)) or 
            (dir_l and self.is_coliding(point_l)) or 
            (dir_u and self.is_coliding(point_u)) or 
            (dir_d and self.is_coliding(point_d)),

            # Danger right r -> d -> l -> u
            (dir_r and self.is_coliding(point_d)) or
            (dir_d and self.is_coliding(point_l)) or 
            (dir_l and self.is_coliding(point_u)) or
            (dir_u and self.is_coliding(point_r)),

            # Danger left
            (dir_r and self.is_coliding(point_u)) or 
            (dir_u and self.is_coliding(point_l)) or 
            (dir_l and self.is_coliding(point_d)) or
            (dir_d and self.is_coliding(point_r)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.apple.x < self.head.x,  # food left
            self.apple.x > self.head.x,  # food right
            self.apple.y < self.head.y,  # food up
            self.apple.y > self.head.y   # food down
        ]

        return np.array(state, dtype=int)
