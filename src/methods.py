import numpy as np
import os
import random

from multiprocessing import Manager
from datetime import datetime

from agent import Agent
from game import Game

def HumanMethod(game, game_options, pygame):
    update_visual(game, pygame)

    if game.restart == True:
        screen = game.screen
        render = game.render
        game_options['track_data'] = game.track
        game = Game(game_options)
        game.screen = screen
        game.render = render
        game.track = game_options['track_data']
        game.restart = False

def SpecificMapMethod(game, pygame, game_options):
    update_visual(game, pygame)
    update_speed(game, pygame)
    if game.environment.agents[0].car.died:
        new_track = random.choice(list(game.tracks.values()))
        game.track = new_track
        game.track_name = [name for name, track in game.tracks.items() if track is game.track][0]
        start_pos, start_dir = game.real_starts[game.track_name]

       
        best_agent, agent = extract_best_agent(f"./data/per_track/{game.track_name}/trained", game_options['environment'], game, start_pos, start_dir)
        load_csv_data("./data/train/log.csv", game)
        print(f" - Loading best agent {best_agent} from {game.track_name}")

        game.environment.agents[0] = agent
        game.environment.generation = best_agent
    
        agent.track = game.track
        game.ticks = 0

def ContinuousMethod(game, game_options, pygame):
    update_visual(game, pygame)
    update_speed(game, pygame)
    continuous_commands(game, pygame)
    if not game.environment.agents[0].car.died: return

    new_track = random.choice(list(game.tracks.values()))
    game.track = new_track
    game.track_name = [name for name, track in game.tracks.items() if track is game.track][0]
    start_pos, start_dir = random.choice(game.start_positions[game.track_name])

    
    best_agent, agent = extract_best_agent("./data/train/trained", game_options['environment'], game, start_pos, start_dir)
    load_csv_data("./data/train/log.csv", game)
    print(f" - Loading best agent: {best_agent} from global training")

    game.environment.agents[0] = agent
    game.environment.generation = best_agent
    agent.track = game.track

    game.ticks = 0

def HumanVSaiMethod(game, game_options, pygame):
    if game.environment.agents[0].car.died:
        game.restart = True
        game.environment.agents[0].car.kill()
        game.environment.agents[0].car.died = False
        game.environment.agents[0].car.x = game.start_pos[0]
        game.environment.agents[0].car.y = game.start_pos[1]
        game.environment.agents[0].car.direction = game.start_dir
    elif game.started:
        game.environment.agents[1].tick(0, game)
        game.environment.agents[0].car.applyPlayerInputs()
        game.environment.agents[0].car.updateCar()
        game.environment.agents[0].car.checkCollisions(game.ticks)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_r] or game.restart == True:
        best_agent, agent = extract_best_agent("./data/train/", game_options['environment'], game, game.start_pos, game.start_dir)
        print(f" - Loading best agent {best_agent}")

        game.environment.agents[1] = agent
        game.environment.agents[0] = Agent(game_options['environment'], game.track, game.start_pos, game.start_dir, game.track_name)
        game.started = False
    if keys[pygame.K_SPACE] and game.started == False:
        game.started = True
        game.environment.agents[0].car.x = game.start_pos[0]
        game.environment.agents[0].car.y = game.start_pos[1]
        game.environment.agents[0].car.direction = game.start_dir
        game.environment.agents[1].car.direction = game.start_dir
        game.environment.agents[1].car.x = game.start_pos[0]
        game.environment.agents[1].car.y = game.start_pos[1]
    if keys[pygame.K_v]:
        game.visual = not game.visual
    
def update_speed(game, pygame):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        game.speed += 1
    if keys[pygame.K_DOWN]:
        game.speed -= 1
    if keys[pygame.K_SPACE]:
        game.speed = 60
    game.speed = max(0, game.speed)

def update_visual(game, pygame):
    keys = pygame.key.get_pressed()
    if game.last_keys_update + 0.3 < datetime.now().timestamp(): 
        if keys[pygame.K_v]:
            game.visual = not game.visual
            game.last_keys_update = datetime.now().timestamp()
        if keys[pygame.K_b]:
            game.debug = not game.debug
            game.last_keys_update = datetime.now().timestamp()

def continuous_commands(game, pygame):
    keys = pygame.key.get_pressed()
    if game.last_keys_update + 0.3 < datetime.now().timestamp():
        if keys[pygame.K_r]:
            game.environment.agents[0].car.kill()
            game.last_keys_update = datetime.now().timestamp()
        
def AgentsRaceMethod(game, game_options, pygame):
    keys = pygame.key.get_pressed()
    if game.last_keys_update + 0.3 < datetime.now().timestamp():
        if keys[pygame.K_v]:
            game.visual = not game.visual
            game.last_keys_update = datetime.now().timestamp()
        if keys[pygame.K_b]:
            game.debug = not game.debug
            game.last_keys_update = datetime.now().timestamp()
        if keys[pygame.K_s]:
            game.environment.agents.append(game.environment.agents.pop(0))
            game.last_keys_update = datetime.now().timestamp()
        if keys[pygame.K_w]:
            game.environment.agents.insert(0, game.environment.agents.pop())
            game.last_keys_update = datetime.now().timestamp()
    if all([agent.car.died for agent in game.environment.agents]):
        for agent in game.environment.agents:
            agent.car.kill()
            agent.car.died = False
            game.ticks = 0

def load_csv_data(path, game):
    try:
        with open(path, "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            game.environment.previous_best_score = float(last_line.split(",")[1])
            game.environment.previous_best_lap = float(last_line.split(",")[2])
    except:
        pass

def extract_best_agent(path, options, game, start_pos, start_dir):
    best_agent = 0
    for file in os.listdir(path):
        if file.endswith(".npy") and not file.endswith("gents.npy"):
            if int(file[11:-4]) > best_agent:
                best_agent = int(file[11:-4])
    agent = Agent(options, game.track, start_pos, start_dir, game.track_name)
    agent.network = np.load(path + "/best_agent_" + str(best_agent) + ".npy", allow_pickle=True).item()['network']
    return best_agent, agent


        
