# Car Neural Network
An attempt at generating the optimal racing line for motorsports, more specifically Formula 1


This is a [TIPE](https://www.scei-concours.fr/tipe.php) for the French "Concours d'ingenieur". Associated documents can be found in `tipe`.

For reference, the final mark obtained after the presentation was 19.3/20 

## Requirements
This code was only meant to run on Apple Silicon machines, and might work on Intel based Macs, although it hasn't been tested. The shaders are pre-compiled in src/shaders, and were compiled locally. If you want to try and compile them on your side, see `make compile_shader`

Nothing here was made to be used, it is just a personal attempt at an interesting problem.

## Some results
__Cup Of The Americas generated racing line__
![image](https://github.com/Al0den/car-neural-network/assets/111601320/59dd49ef-b57f-48d8-8f03-62820887eb74)
![image](https://github.com/Al0den/car-neural-network/assets/111601320/d190e44d-b929-41b1-a346-fd81e38e7ad0)

__Mexico Grand Prix generated racing line__
![image](https://github.com/Al0den/car-neural-network/assets/111601320/717f3e9b-1d57-4052-bc21-84407aedc549)
![image](https://github.com/Al0den/car-neural-network/assets/111601320/8b88b6d7-cb06-4fcb-882f-152dc935cf7b)

## Trying the code out
Some already trained agents are by default in the code (Lots are in reality, stored in `data/train/trained`), and should work just fine when cloning the repo.

To run pre-trained agent on a random start position, simpy run `python src/main.py`, and then press `5`

To get a specific trained agent run, use `4`, and to get the best one select agent number `0`

**Modes**:

`0` - Playing as a Human on track

`1` - Create a set of untrained agents

`2` - Continue the training of the agent batch

`3` - Train only on a specific map

`4` - View run on a specific map

`5` - View global run

`6` - Race against the AI

`7` - Generate the optimal track, that is output in `data/per_track/{track_name}`

`8` - Show multiple agents

`9` - \

`10` - Performance Test


## Configuration

In `src/settings.py`, most of the configuration options are available. Some may be outdated/un-used, and changing some of them makes the code extremely unstable/unrealistic



