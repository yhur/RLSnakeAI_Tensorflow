Snake Game Transfer Learning for SnakeGame RL
---
This commit level shows the Transfer Learning for SnakeGame RL.

Assume you trained the Snake with this states with the tag `model.fit` of this repository.
```json
  state = [
      # Danger straight - Wall
      self.isWallAhead(forward),
      # Danger right - Wall
      self.isWallAhead(right),
      # Danger left - Wall
      self.isWallAhead(left),
      # current moving direction as one hot encoding [R, D, L, U]
      *[self.direction == d for d in list(Direction)],
      # Danger straight - body
      self.bodyCheck(forward),
      self.bodyCheck(right),
      self.bodyCheck(left),
      # Food location 
      self.board.apple.x < self.head.x,  # food left
      self.board.apple.x > self.head.x,  # food right
      self.board.apple.y < self.head.y,  # food up
      self.board.apple.y > self.head.y   # food down
  ]
```

And you want to add three more input values regarding if it has at least one safe move from new position like this. Since the Neural Network is different, you need to train from scratch. But can we transfer the learning from above Snake to the new Snake?

```json
  state = [
      # Danger straight - Wall
      self.isWallAhead(forward),
      # Danger right - Wall
      self.isWallAhead(right),
      # Danger left - Wall
      self.isWallAhead(left),
      # current moving direction as one hot encoding [R, D, L, U]
      *[self.direction == d for d in list(Direction)],
      # Danger straight - body
      self.bodyCheck(forward),
      self.bodyCheck(right),
      self.bodyCheck(left),
      # Food location 
      self.board.apple.x < self.head.x,  # food left
      self.board.apple.x > self.head.x,  # food right
      self.board.apple.y < self.head.y,  # food up
      self.board.apple.y > self.head.y,   # food down
      self.isMoveSafe(forward),
      self.isMoveSafe(right),
      self.isMoveSafe(left)
  ]
```

How to excercise
1. clone the repository
2. checkout to model.fit
3. train it
4. checkout to `transfer_learning`
5. rename model to org
6. run `python transfer_game.py`
7. once finished, you can run `python game_tf.py show`


```sh
Usage: transfer_game.py [OPTIONS] [CMD]

  Ex)python transfer_game.py -m model 

Options:
  -o, --org   TEXT            Original model File
  -m, --model TEXT            Stored model File
  -s, --speed INTEGER         pygame speed
  -w, --width INTEGER         board width
  -b, --board_height INTEGER  board height
  -t, --transfer_num INTEGER  number of transfer learning
  -v, --verbose               Enable verbose mode.
  -h, --help                  Show this message and exit.             Show this message and exit.
```
