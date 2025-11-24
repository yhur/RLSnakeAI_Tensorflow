Snake Game Reinforcement Learning with Tensorflow
```sh
Usage: game_tf.py [OPTIONS] [CMD]

  Ex)python game_tf.py -m model/model.weights.h5 -s 500 -w 32 -b 24 show

Options:
  -m, --model TEXT            Stored model File
  -s, --speed INTEGER         pygame speed
  -w, --width INTEGER         board width
  -b, --board_height INTEGER  board height
  -v, --verbose               Enable verbose mode.
  -h, --help                  Show this message and exit.             Show this message and exit.
```

This branch demonstrates Transfer Learning for the case when a new kind of data is added to the input layer, specifically move safety for each direction in order to avoid self-traps.

To run this lab:

* First, pull either the model.fit or snake_distance tag commit of the repo
* Run it until it accumulates some level of skill using python -m newmodel show
* Checkout to this branch
* Run migrate.ipynb to migrate the weights of the existing model to a new model. This sets all additional weights of the first hidden layer, padding them with zeros.
* Then run python -m newmodel show