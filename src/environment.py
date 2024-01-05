import numpy as np
import random
import os

from agent import Agent
from utils import copy_network
from settings import *


class Environment:
    def __init__(self, options, track, player, start_pos, start_dir, track_name=None):
        self.options = options
        self.track = track
        self.alive = self.options['num_agents']
        self.generation = self.options['generation_to_load']
        self.player = player
        
        self.start_pos = start_pos
        self.start_dir = start_dir

        self.previous_best_lap = 0
        self.previous_best_score = 0

        self.agents = np.array([Agent(options, track, start_pos, start_dir, track_name) for _ in range(self.options['num_agents'])])
        
        if player == 2 or player == 1: self.previous_agents = [None] * 30
        else: self.previous_agents = [None] * 10

    def next_generation(self, game):
        game.track_name = random.choice(list(game.tracks.keys()))
        game.track = game.tracks[game.track_name]

        self.track = game.track
        self.start_pos, self.start_dir = random.choice(game.start_positions[game.track_name])

        for agent in self.agents:
            if agent.car.laps != game.map_tries:
                agent.car.lap_time = 0
            else:
                agent.car.lap_time = -agent.car.lap_time

        self.agents = sorted(self.agents, key=lambda x: (x.car.laps, x.car.lap_time, x.car.score), reverse=True)

        for agent in self.agents:
            agent.car.lap_time = -agent.car.lap_time
        ranked_agents = self.agents

        scores = [agent.car.score for agent in ranked_agents]

        if ranked_agents[0].car.laps == game.map_tries:
            self.previous_best_lap = ranked_agents[0].car.lap_time
        else:
            self.previous_best_lap = 0
        self.previous_best_score = max([agent.car.score for agent in ranked_agents])
    
        best_agents = ranked_agents[:int(len(ranked_agents) * 0.01) + 1]
        new_agents = []
        for father in best_agents:
            child = Agent(self.options, self.track, self.start_pos, self.start_dir, game.track_name)
            child.network = copy_network(father.network)
            child.evolution = father.evolution + ["i"]
            child.mutation_rates = father.mutation_rates + ["-"]
            new_agents.append(child)
        for _ in range(int(len(ranked_agents) * previous_ratio)):
            if self.previous_agents[0] == None: break
            child = Agent(self.options, self.track, self.start_pos, self.start_dir, game.track_name)
            father = None
            while father == None:
                father = self.linear_weighted_selection(self.previous_agents, 1)
            if father != None:
                child.network = copy_network(father.network)
                child.evolution = father.evolution + ["p"]
                child.mutation_rates = father.mutation_rates + ["-"]
                new_agents.append(child)
       
        while len(new_agents) < len(self.agents):
            randint = np.random.uniform(0,1)
            child = Agent(self.options, self.track, self.start_pos, self.start_dir, game.track_name)
            if randint > 1 - only_mutate_rate:
                father = self.linear_weighted_selection(ranked_agents)
                child = Agent(self.options, self.track, self.start_pos, self.start_dir, game.track_name)
                child.network = copy_network(father.network)
                rate = self.mutate(child)
                child.evolution = father.evolution + ["m"]
                child.mutation_rates = father.mutation_rates + [rate]
            elif randint > 1 - (cross_over_rate + only_mutate_rate):
                father = self.linear_weighted_selection(ranked_agents)
                mother = self.linear_weighted_selection(ranked_agents)
                child = self.crossover(father, mother, game)
                new_rand = np.random.uniform(0,1)
                if new_rand > 1 - mutate_cross_over_rate: # Most cross-overs are mutated
                    rate = self.mutate(child)
                    child.evolution = father.evolution + ["n"]
                    child.mutation_rates = father.mutation_rates + [rate]
                else:
                    child.evolution = father.evolution + ["c"]
                    child.mutation_rates = father.mutation_rates + ["-"]
            else:
                child.evolution = child.evolution + ["-"] * self.generation + ["r"]
                child.mutation_rates = child.mutation_rates + ["-"]
            new_agents.append(child)

        if game.player != 3:
            self.log_data(ranked_agents)
            self.save_agents()
            self.save_best_agent(ranked_agents[0], self.generation)
        else:
            self.save_agents(f"./data/per_track/{game.track_name}/trained/agents")
            self.log_data(ranked_agents, f"./data/per_track/{game.track_name}/log.csv")
            self.save_best_agent(ranked_agents[0], self.generation, f"./data/per_track/{game.track_name}/trained/best_agent_")

        self.add_previous_best(ranked_agents[0], game)

        # Mix up agents to prevent getting best agents in first indices
        random.shuffle(new_agents)

        self.agents = new_agents
        self.generation += 1
        self.alive = len(self.agents)
    
    def linear_weighted_selection(self, ranked_agents, coeff=7):
        num_agents = len(ranked_agents)
        
        selection_weights = [num_agents - i + 1 for i in range(num_agents)]
        selection_weights = [i ** coeff for i in selection_weights]
        indices = [i for i in range(num_agents)]
        selected_agent = random.choices(indices, weights=selection_weights)[0]
        selected_agent = ranked_agents[selected_agent]
        
        return selected_agent

    def save_agents(self, path="./data/train/agents"):
        for agent in self.agents:
            agent.car.track = []
            agent.car.previous_points = []
        data = {
            'agents': self.agents,
            'previous_agents': self.previous_agents,
            'input_size': self.options['state_space_size'],
            'output_size': self.options['action_space_size'],
            'hidden_layer_size': self.options['hidden_layer_size'],
            'num_hidden_layers': self.options['num_hidden_layers'],
            'generation': self.generation
        }
        np.save(path, data, allow_pickle=True)

    def load_agents(self):
        data = np.load("./data/train/agents.npy", allow_pickle=True).item()
        self.agents = data['agents']
        self.previous_agents = data['previous_agents']
        self.options['state_space_size'] = data['input_size']
        self.options['action_space_size'] = data['output_size']
        self.options['hidden_layer_size'] = data['hidden_layer_size']
        self.options['num_hidden_layers'] = data['num_hidden_layers']
        self.generation = data['generation'] + 1 # Since we saved agents, we are starting to teach them the next generation

        for agent in self.agents:
            agent.car.speed = 0
            agent.car.direction = agent.car.start_direction
            agent.car.x = agent.car.start_x
            agent.car.y = agent.car.start_y
            agent.car.score = 0
            agent.car.died = False
            agent.car.checkpoints_seen = []
            agent.last_update = 0

        if (self.player != 2 and self.player != 1):
            self.previous_agents = self.previous_agents[:10]

        print(f" * Loaded all agents from file, generation: {str(self.generation)}")

    def save_best_agent(self, best_agent, generations, path="./data/train/trained/best_agent_"):
        data = {
            'network': best_agent.network,
            'input_size': self.options['state_space_size'],
            'output_size': self.options['action_space_size'],
            'hidden_layer_size': self.options['hidden_layer_size'],
            'num_hidden_layers': self.options['num_hidden_layers']
        }
        np.save(path + str(generations), data)
    
    def crossover(self, parent1, parent2, game):
        child = Agent(self.options, self.track, self.start_pos, self.start_dir, game.track_name)
        child.network = copy_network(parent1.network)
        for i in range(len(child.network)):
            for j in range(len(child.network[i])):
                if np.random.uniform(0, 1) > 0.5:
                    child.network[i][j] = parent2.network[i][j]
        return child
    
    def mutate(self, agent):
        rate = np.random.choice(mutation_rates)
        agent.Mutate(rate)
        return rate

    def log_data(self, ranked_agents, path="./data/train/log.csv"):
        if not os.path.isfile(path):
            with open(path, "w") as file:
                file.write("Generation, Best Score, Average Score, Best Lap Time, Laps, Average Laps, Number of Neurons, Best Agent Evolution, Mutation Rates Used\n")
        best_lap_time = self.previous_best_lap
        best_score = self.previous_best_score/ (map_tries)
        best_agent_evolution = "".join(ranked_agents[0].evolution)
        best_agent_rates = "/".join([str(rate) for rate in ranked_agents[0].mutation_rates])
        generation = self.generation
        laps = ranked_agents[0].car.laps
        average_score = np.average([agent.car.score for agent in ranked_agents]) / map_tries
        average_lap = np.average([agent.car.laps for agent in ranked_agents])

        neurons = 0
        for layer in ranked_agents[0].network:
            neurons += layer.shape[0] * layer.shape[1]

        with open(path, "a") as file:
            file.write(f"{generation}, {best_score}, {average_score}, {best_lap_time}, {laps}, {average_lap}, {neurons}, {best_agent_evolution}, {best_agent_rates}\n")

    def load_specific_agents(self, game):
        method = "global"
        try:
            method = "specialised"
            data = np.load("./data/per_track/" + game.track_name + "/trained/agents.npy", allow_pickle=True).item()
        except:
            method = "global"
            data = np.load("./data/train/agents.npy", allow_pickle=True).item()
            os.makedirs("./data/per_track/" + game.track_name + "/trained/", exist_ok=True)
        self.agents = data['agents']
        for _ in range(int(specialised_training_multiple) - 1):
            if method == "specialised": continue
            new_data = np.load("./data/train/agents.npy", allow_pickle=True).item()
            self.agents += new_data['agents']
        self.options['state_space_size'] = data['input_size']
        self.options['action_space_size'] = data['output_size']
        self.options['hidden_layer_size'] = data['hidden_layer_size']
        self.options['num_hidden_layers'] = data['num_hidden_layers']
        self.generation = data['generation'] + 1

        for agent in self.agents:
            agent.car.speed = 0
            agent.car.direction = agent.car.start_direction
            agent.car.x = agent.car.start_x
            agent.car.y = agent.car.start_y
            agent.car.score = 0
            agent.car.died = False
            agent.car.checkpoints_seen = []
            agent.last_update = 0

    
        print(f" - Loaded all agents from {method} training, generation: {str(self.generation)}")

    def add_previous_best(self, best_agent, game):
        if self.previous_agents[0] == None or (self.previous_agents[0].car.lap_time > best_agent.car.lap_time and best_agent.car.laps == game.map_tries) or game.player in [1, 2]:
            self.previous_agents.insert(0, best_agent)
            self.previous_agents.pop()
        elif self.previous_agents[0].car.lap_time == 0: # - If no previous best lap time, add the first agent as it is just used as safety
            self.previous_agents.insert(0, best_agent)
            self.previous_agents.pop()

    def get_agent_network(self, path):
        data = np.load(path, allow_pickle=True).item()
        return data['network']
