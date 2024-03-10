import os
import sys
import time

import numpy as np

from load_options import load_options
from methods import HumanMethod, SpecificMapMethod, ContinuousMethod, HumanVSaiMethod, AgentsRaceMethod
from game import Game
from settings import *
from utils import SaveOptimalLine, SaveAgentsSpeedGraph, InitialiseDisplay

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

def main(): 
    os.makedirs("./data/tracks", exist_ok=True)
    os.makedirs("./data/train/trained", exist_ok=True)
    os.makedirs("./data/per_track", exist_ok=True)
    game_options = load_options()
    game = Game(game_options)

    if god: print("WARNING: God mode is on, car will ignore track boundaries")
    if game.debug: print("WARNING: Debug is on, console can get spammed")
    if not pre_load: print("WARNING: Pre load is not on, track center-line will be calculated at every step")

    if game_options['display']:
        import pygame
        clock, last_render = InitialiseDisplay(game, game_options, pygame)
    
    no_tick = False

    if game.player == 7:
        generated = np.copy(game.track).astype(np.uint16)
        brake, steer, throttle, speeds, timestamps = [], [], [], [], []
    game.restart = False

    # Main loop
    while game.running.value:
        if game_options['display']:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    game.running.value = False
        
        if not no_tick: value = game.tick()
        no_tick = False

        if game.player == 0:
            clock.tick(game.speed)
            game.agent = False
            game.render.RenderFrame(game)
            HumanMethod(game, game_options, pygame)
        elif game.player == 4:
            game.speed = max(0, game.speed)
            clock.tick(game.speed)
            SpecificMapMethod(game, pygame, game_options)
            game.render.RenderFrame(game)
            pygame.display.update()
        elif game.player == 5:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_p]: no_tick = True
            else:
                game.speed = max(0, game.speed)
                clock.tick(game.speed)
                ContinuousMethod(game, game_options, pygame)
                if time.time() - last_render < 0.015: continue
                last_render = time.time()
                game.render.RenderFrame(game)
        elif game.player == 6:
            keys = pygame.key.get_pressed()
            HumanVSaiMethod(game, game_options, pygame)
            game.render.RenderFrame(game)
            clock.tick(game.speed)
        elif game.player == 7:
            x, y = game.environment.agents[0].car.x, game.environment.agents[0].car.y
            generated[int(y), int(x)] = 10 + game.environment.agents[0].car.speed
            
            speeds.append(game.environment.agents[0].car.speed)
            throttle.append(game.environment.agents[0].car.acceleration)
            brake.append(game.environment.agents[0].car.brake)
            steer.append(game.environment.agents[0].car.steer)
            timestamps.append(game.ticks * delta_t)
            if game.environment.agents[0].car.died:
                print(" * Saving generated track...")
                SaveOptimalLine(generated, game.track_name, game.best_agent)
                SaveAgentsSpeedGraph(speeds, throttle, brake, game.best_agent, game.track_name, steer)
                print(f" * Saved generated track to ./data/per_track/{game.track_name}/generated_{game.best_agent}.png")
                game.running.value = False
                run_data = {
                    "timestamps": timestamps,
                    "speeds": speeds,
                    "throttle": throttle,
                    "brake": brake,
                    "steer": steer,
                }
                np.save(f"./data/per_track/{game.track_name}/generated_data.npy", run_data)
        elif game.player == 8:
            AgentsRaceMethod(game, game_options, pygame)
            game.render.RenderFrame(game)
            clock.tick(game.speed)
        elif game.player == 9:
            if not value: game.running.value = False
            data = { 
                "info": game.generated_data,
                "agents": game.environment.agents,
            }
            np.save(f"./data/per_track/{game.track_name}/generated_run.npy", data)
    if game_options['display']:
        pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

