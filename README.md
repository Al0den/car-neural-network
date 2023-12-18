# Car Neural Network
This repo is a storing repo for my TIPE project in French Classe Preparatoire. It has the objective to train a neural network to follow a specific track, and learn the fastest way to complete a full lap on it.
In no means is it perfect or optimal, but a learning curve and a work in progress

## Using the code
Currently, this wasnt made to be run or used. However, nothing should prevent it to work on another machine, given a correct track (5000x5000 image, stored in /data/tracks). The image should contain black representing the track, orange for checkpoints and red for finish line. start\_x and start\_y need to be manually adjusted in `src/car.py` manually. 

By default, it is only setup for singapoure, which is provided

Simply run `python src/main.py` and follow the prompts to enter the desired mode

