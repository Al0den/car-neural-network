import gc
import os
import numpy as np
import random
import time
import json
import threading

from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from multiprocessing import Process, Array, Manager, Value

Image.MAX_IMAGE_PIXELS = None

from environment import Environment
from car import Car
from utils import is_color_within_margin, get_new_starts, get_nearest_centerline, update_terminal
from agent import Agent
from settings import *
from smoothen import smoothen
from metal import Metal
from corners import get_corners

from utils import next_speed

class Game:
    def __init__(self, game_options):
        self.player = game_options['player']
        self.screen_width = game_options['screen_width']
        self.screen_height = game_options['screen_height']
        self.display = game_options['display']
        self.options = game_options

        self.ticks = 0 # Ticks counter (Should be agent specific ideally)
        self.speed = base_game_speed # Game tps
        
        self.restart = False # If the game needs to be restarted, set this to True
        self.started = False # Only useful for player == 6, defines if race has started
        self.running = Value('b', True) # If the game needs to be stopped, set this to False to kill main loop

        self.map_tries = map_tries # - Number of total tries
        self.mutation_strength = game_options['mutation_strength']

        self.screen = None # Display parameter
        self.render = None # Display parameter
        self.clock = None # Display parameter
        self.visual = True # - Display parameter
        self.debug = debug # - Display parameter
        self.last_keys_update = 0 # - Time since last key click, Display parameter
        self.track_name = game_options['track_name'] # - Track name

        self.pause = False

        self.shared_tracks = None
        self.speed_precalc = None

        with open("./src/config.json") as f: self.config = json.load(f)

        if self.player in single_track_players: self.load_single_track()
        else: self.load_all_tracks()

        self.generating_corners = False
        self.generated_corners = []
        self.generated_starts = []
        self.generated_start_tracks = []
        
        self.environment = Environment(game_options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)
        
        self.car_numbers = [0] * len(self.environment.agents)

        self.issues = Value('i', 0, lock=False)
        
        if self.player == 0:
            self.car = Car(self.track, self.start_pos, self.start_dir, self.track_name)
            self.car.setFutureCorners(self.corners[self.track_name])
        elif self.player == 2:
            self.environment.load_agents(self)
            self.player = 1
        elif self.player == 3:
            self.environment.load_specific_agents(self)
        elif self.player == 4:
            if self.options["generation_to_load"] == 0:
                best_agent, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained")
            else:
                best_agent = self.options["generation_to_load"]
                agent = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                agent.network = np.load(f"./data/per_track/{self.track_name}/trained/best_agent_{best_agent}.npy", allow_pickle=True).item()['network']
            self.extract_csv(f"./data/per_track/{self.track_name}/log.csv")
            agent.car.track = self.track
            agent.car.track_name = self.track_name
            self.environment.generation = best_agent
            self.environment.agents[0] = agent
            
            self.environment.agents[0].car.speed = self.config['quali_start_speed'].get(self.track_name)
            self.environment.agents[0].car.acceleration = 1
            self.environment.agents[0].car.setFutureCorners(self.corners[self.track_name])

            
        elif self.player == 5:
            best_agent, agent = self.load_best_agent("./data/train/trained")
            self.extract_csv("./data/train/log.csv")
            agent.car.track = self.track
            agent.car.track_name = self.track_name
            agent.car.setFutureCorners(self.corners[self.track_name])

            self.environment.generation = best_agent
            self.environment.agents[0] = agent
        elif self.player == 6:
            import pygame
            _, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained/")

            self.environment.agents[1] = agent
            self.environment.agents[0] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)

            self.environment.agents[1].car.speed = self.config['quali_start_speed'].get(self.track_name)
            self.environment.agents[1].car.acceleration = 1

            self.environment.agents[0].car.speed = self.config['quali_start_speed'].get(self.track_name)
            self.environment.agents[0].car.acceleration = 1

            self.environment.agents[1].car.setFutureCorners(self.corners[self.track_name])
            self.environment.agents[0].car.setFutureCorners(self.corners[self.track_name])

            self.data = [{"speed": [], "power": [], "brake": []}, {"speed": [], "power": [], "brake": []}]

        elif self.player == 7:
            
            if game_options['generation_to_load'] == 0:
                best_agent, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained/")
            else:
                best_agent = game_options['generation_to_load']
                agent = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                agent.network = np.load("./data/per_track/" + self.track_name + "/trained/best_agent_" + str(best_agent) + ".npy", allow_pickle=True).item()['network']
            agent.car.speed = self.config['quali_start_speed'].get(self.track_name)
            agent.car.acceleration = 1
            agent.car.setFutureCorners(self.corners[self.track_name])
            

            self.best_agent = best_agent
            self.environment.agents[0] = agent
            
        elif self.player == 8:
            self.s_or_g_choice = self.options["s_or_g"]

            if self.s_or_g_choice == "g":
                start_generation = np.load("./data/train/agents.npy", allow_pickle=True).item()['generation']
                gap = self.options["gap"]
                print(f" - Selecting agents: {[start_generation - i * gap for i in range(len(self.environment.agents))]}")
                # Check that for every i from 0-num_agents - 1, agents.current_generation - i * gap file exists
                if start_generation - gap * len(self.environment.agents) <= 0:
                    self.exit("Incorrect agent values")
                self.environment.agents = [None] * game_options['environment']['num_agents']
                corners = None
                for i in range(len(self.environment.agents)):
                    self.environment.agents[i] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                    gen_num = start_generation - i * gap
                    network = np.load(f"./data/train/trained/best_agent_{gen_num}.npy", allow_pickle=True).item()['network']
                    self.environment.agents[i].network = network
                    self.environment.agents[i].car.speed = 0
                    if corners == None:
                        corners = self.environment.agents[i].car.setFutureCorners(self.corners[self.track_name])
                    else:
                        self.environment.agents[i].car.future_corners = np.copy(corners).tolist()

                self.environment.agents = self.environment.agents[::-1]
                initial = start_generation - gap * len(self.environment.agents)
                self.car_numbers = [initial + i * gap for i in range(len(self.environment.agents))]
            else:
                if not os.path.exists(f"./data/per_track/{self.track_name}/log.csv"):
                    self.exit("No specific trained agent found, exiting...")
                agent_nums = []
                lap_times = []
                with open(f"./data/per_track/{self.track_name}/log.csv", "r") as f:
                    lines = f.readlines()
                    lines = lines[2:]
                    for line in lines:
                        if line.split(",")[3] not in lap_times:
                            lap_times.append(line.split(",")[3])
                            agent_nums.append(int(line.split(",")[0]))
                agent_nums.reverse()
                if agent_nums == []: self.exit("Not enough agents available, exiting...")
                # Only keep the game_options['envirocnment']['num_agents'] best agents
                agent_nums = agent_nums[:game_options['environment']['num_agents']]
                self.environment.agents = [None] * len(agent_nums)
                self.car_numbers = [0] * len(agent_nums)
                corners = None
                for i in range(len(agent_nums)):
                    self.car_numbers[i] = agent_nums[i]
                    self.environment.agents[i] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                    self.environment.agents[i].network = np.load(f"./data/per_track/{self.track_name}/trained/best_agent_{agent_nums[i]}.npy", allow_pickle=True).item()['network']
                    self.environment.agents[i].car.speed = self.config['quali_start_speed'].get(self.track_name)
                    self.environment.agents[i].car.acceleration = 1
                    if corners == None:
                        corners = self.environment.agents[i].car.setFutureCorners(self.corners[self.track_name])
                    else:
                        self.environment.agents[i].car.future_corners = np.copy(corners).tolist()
        elif self.player == 9:
            self.generated_data = []
            available, lap_times, possible_agent_number, possible_agent_laps = [], [], [], []

            with open("./data/per_track/" + self.track_name + "/log.csv", "r") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    if line.split(",")[2] not in possible_agent_laps:
                        lap_times.append(line.split(",")[3])
                        available.append(line.split(",")[0])
                        possible_agent_laps.append(line.split(",")[2])
                
            possible_agent_number = possible_agent_number[-game_options['environment']['num_agents']:]
            for i in range(len(possible_agent_number)):
                self.environment.agents[i] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                self.environment.agents[i].network = np.load("./data/per_track/" + self.track_name + "/trained/best_agent_" + str(possible_agent_number[i]) + ".npy", allow_pickle=True).item()['network']
                self.environment.agents[i].car.speed = self.options['quali_start_speed'].get(self.track_name)
                self.environment.agents[i].car.acceleration = 1
            self.environment.agents = self.environment.agents[::-1]
            available.sort(reverse=True)
            lap_times.reverse()
            lap_times = lap_times[:len(self.environment.agents)]
            self.lowest_lap_time = min(lap_times)
            available = available[:len(self.environment.agents)]
            for i in range(len(self.environment.agents)):
                self.environment.agents[i].network = self.environment.get_agent_network(f"./data/per_track/{self.track_name}/trained/best_agent_{available.pop()}.npy")
                self.generated_data.append([])
                agent = self.environment.agents[i]
                agent.car.x = self.real_starts[self.track_name][0][1]
                agent.car.y = self.real_starts[self.track_name][0][0]
                agent.car.start_x = self.real_starts[self.track_name][0][1]
                agent.car.start_y = self.real_starts[self.track_name][0][0]
                agent.car.direction = self.real_starts[self.track_name][1]
                agent.car.start_direction = self.real_starts[self.track_name][1]
                assert(agent.car.track[agent.car.y, agent.car.x] == 10)
        elif self.player == 10:
            print(" - Starting performance test...")
            self.totalScore = 0
            self.runs = 0
            self.pre_calc_speed = []
            for i in range(0, 3399, 1):
                self.pre_calc_speed.append(next_speed(i/10))
            self.pre_calc_speed = np.array(self.pre_calc_speed)

            for i in range(len(self.environment.agents)):
                self.environment.agents[i].car.speed_pre_calc = self.pre_calc_speed
            
        elif self.player == 11:
            data = np.load("./data/per_track/" + self.track_name + "/generated_run.npy", allow_pickle=True).item()
            self.environment.agents = data['agents']
            self.generated_data = data['info']
            for agent in self.environment.agents:
                agent.car.x = self.real_starts[self.track_name][0][1]
                agent.car.y = self.real_starts[self.track_name][0][0]
                agent.car.direction = self.real_starts[self.track_name][1]
                agent.car.track = self.track
                assert(agent.car.track[agent.car.y, agent.car.x] == 10)

        if self.player in [1, 2, 3]:
            self.initialise_process()
            print(" - Reloading tracks...")
            if self.player == 3:
                self.load_single_track()
            else:
                self.load_all_tracks()

        if self.player in global_metal_instance_players:
            self.Metal = Metal(self.tracks)
            self.track_index = self.Metal.getTrackIndexes()
            num_agents = len(self.environment.agents)
            self.Metal.init_shaders(len(points_offset) * 10 * num_agents, len(points_offset) * num_agents, points_offset, num_agents * 8, num_agents * 4)
        
        if self.player == 8:
            self.multiple_players_shader_init()

        self.prev_update = time.time()
        self.start_time = time.time()

        self.CalculateProgressionValues()
    
    def tick(self):
        if self.pause: return
        if self.player == 0:
            self.ticks += 1
            if self.ticks % 100 == 0 and debug:
                print(f"Ticks: {self.ticks}, score {self.car.ScoreCar(self.track_scores[self.track_name])}. max_potential: {self.car.MaxPotential(self.track_scores[self.track_name])}")
            self.car.Tick(self)
            if self.car.died == True:
                self.restart = True
                self.car.died = False
                if self.debug: print("Car died, restarting")
        elif self.player == 1 or self.player == 3:
            self.train_agents_gpu()
        elif self.player == 4:
            speed = self.environment.agents[0].car.speed
            brake = self.environment.agents[0].car.brake
            throttle = self.environment.agents[0].car.acceleration
            steer = self.environment.agents[0].car.steer

            self.render.GraphData(speed, brake, throttle, steer, self.ticks)

            self.environment.agents[0].Tick(self.ticks, self)
            if self.environment.agents[0].car.died == True:
                self.render.ClearData()
            self.ticks += 1
        elif self.player == 5:
            import pygame
            keys = pygame.key.get_pressed()
            speed = self.environment.agents[0].car.speed
            brake = self.environment.agents[0].car.brake
            throttle = self.environment.agents[0].car.acceleration
            steer = self.environment.agents[0].car.steer

            self.render.GraphData(speed, brake, throttle, steer, self.ticks)

            if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_h]:
                self.environment.agents[0].car.ApplyPlayerInputs()
                self.environment.agents[0].car.UpdateCar()
                self.environment.agents[0].car.CheckCollisions(self.ticks)
            else:
                self.environment.agents[0].Tick(self.ticks, self)
                if self.environment.agents[0].car.died == True:
                    self.render.ClearData()
            self.ticks += 1
        elif self.player == 6:
            if self.started:
                self.environment.agents[0].car.ApplyPlayerInputs()
                self.environment.agents[0].car.UpdateCar()
                self.environment.agents[0].car.CheckCollisions(self.ticks)
                self.environment.agents[1].Tick(self.ticks, self)
                self.ticks += 1
                if self.environment.agents[0].car.speed == 0:
                    # Just copy the previous value in all arrays
                    self.data[0]["speed"].append(self.data[0]["speed"][-1])
                    self.data[0]["power"].append(self.data[0]["power"][-1])
                    self.data[0]["brake"].append(self.data[0]["brake"][-1])
                    self.data[1]["speed"].append(self.data[1]["speed"][-1])
                    self.data[1]["power"].append(self.data[1]["power"][-1])
                    self.data[1]["brake"].append(self.data[1]["brake"][-1])
                else:
                    self.data[0]["speed"].append(self.environment.agents[0].car.speed)
                    self.data[0]["power"].append(self.environment.agents[0].car.acceleration)
                    self.data[0]["brake"].append(self.environment.agents[0].car.brake)
                    self.data[1]["speed"].append(self.environment.agents[1].car.speed)
                    self.data[1]["power"].append(self.environment.agents[1].car.acceleration)
                    self.data[1]["brake"].append(self.environment.agents[1].car.brake)

        elif self.player == 7:
            self.environment.agents[0].Tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 8:
            count = 0
            while self.environment.agents[0].car.died and count < 100:
                self.environment.agents.append(self.environment.agents.pop(0))
                self.car_numbers.append(self.car_numbers.pop(0))
            for i in range(len(self.environment.agents)):
                agent = self.environment.agents[i]
                if agent.car.died: 
                    self.Metal.inVectorBuffer[i * 10] = -1
                    continue
                index = i * 10
                self.Metal.inVectorBuffer[index:index + 3] = [agent.car.int_x, agent.car.int_y, agent.car.int_direction]
                index = i * 8
                self.Metal.cornerBuffer[index:index+4] = agent.car.future_corners[0]
                self.Metal.cornerBuffer[index+4:index+8] = agent.car.future_corners[1]
              
            self.Metal.pointsAndCorner(len(self.environment.agents) * len(points_offset), len(self.environment.agents) * 2)
            per_agents_points = self.Metal.outVectorBuffer.reshape((len(self.environment.agents), len(points_offset)))
            per_agents_corner = self.Metal.cornerOutBuffer.reshape(len(self.environment.agents), 4)

            for i in range(len(self.environment.agents)):
                if self.environment.agents[i].car.died == True: continue
                self.environment.agents[i].Tick(self.ticks, self, per_agents_points[i], per_agents_corner[i])
            self.ticks += 1
        elif self.player == 9:
            any_alive = False
            for i in range(len(self.environment.agents)):
                if self.environment.agents[i].car.died == True: continue
                any_alive = True
                self.environment.agents[i].Tick(self.ticks, self)
                car = self.environment.agents[i].car
                self.generated_data[i].append([car.x, car.y, car.direction])
            self.ticks += 1
            print(f"Currently on tick: {self.ticks} /{self.lowest_lap_time}\r", end='', flush=True)
            return any_alive
        elif self.player == 10:
            self.performanceTest()
            # If more than 2s has passed since last update, update the screen
            if time.time() - self.prev_update > 2:
                self.prev_update = time.time()
                print(f"Average time/tick: {self.totalScore / self.runs:.5f}ms, ran {self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks} ticks")
            if time.time() - self.start_time > perft_duration:
                print(f"Average tps: {(self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks / perft_duration):.2f}, real speed x{(self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks / (perft_duration * 1/delta_t)):.2f}")
                self.running.value = False
        elif self.player == 11:
            any_alive = False
            for i in range(len(self.environment.agents)):
                agent = self.environment.agents[i]
                if agent.car.died == True: continue
                any_alive = True
                data = self.generated_data[i].pop(0)
                agent.car.x = data[0]
                agent.car.y = data[1]
                agent.car.direction = data[2]
            self.ticks += 1

    def initialise_process(self):
        num_processes = self.options['cores']
        num_agents = len(self.environment.agents)
        processes = []

        gc.collect()

        self.agents_feed = Manager().list()
        self.main_lock = Manager().Lock()
        self.lap_times = Array('i', [0] * num_agents, lock=False)
        self.laps = Array('i', [0] * num_agents, lock=False)
        self.scores = Array('d', [0] * num_agents, lock=False)
        self.working = Array('i', [0] * num_processes, lock=False)
        self.log_data = Array('i', [0] * num_processes * 5, lock=False)
        self.max_potentials = Array('d', [0] * self.map_tries, lock=False)
        self.deaths_per_agent = Array('i', [0] * num_agents, lock=False)
        self.threshold = Value('i', 6, lock=False)

        for i in range(num_processes):
            process = Process(target=self.create_process, args=(self.agents_feed, self.log_data, self.scores, self.laps, self.lap_times, self.working, i, self.main_lock, self.max_potentials, self.deaths_per_agent, self.threshold))
            processes.append(process)
        
        for i in range(len(processes)):
            print(f" - Starting process {i+1}/{len(processes)}         \r", end='', flush=True)
            processes[i].start()
        print(f" * Started {len(processes)} processes, running {len(self.environment.agents) * self.map_tries} agents in total, on {len(self.track_names)} tracks")
    
    def create_process(self, agents_feed, log_data, scores, laps, lap_times, working, p_id, main_lock, max_potentials, deaths_per_agent, threshold):
        local_scores = np.array([0] * len(self.environment.agents))
        local_laps = np.array([0] * len(self.environment.agents))
        local_lap_times = np.array([0] * len(self.environment.agents))

        MetalInstance = Metal(self.tracks)
        track_index = MetalInstance.getTrackIndexes()

        num_agents = len(self.environment.agents)

        self.tracks = {}
        
        speed_pre_calc = [0] * 3601
        for i in range(0, 3399, 1):
            speed_pre_calc[i] = next_speed(i/10)

        speed_pre_calc = np.array(speed_pre_calc)

        for track_name in self.track_names:
            track_id = track_index[track_name]
            self.tracks[track_name] = MetalInstance.GetTrackFromBuffer(track_id)

        while True:
            try:
                if working[p_id] == 4: continue
                input_feed = agents_feed.pop(0)
                working[p_id] = 1
            except:
                time.sleep(0.5)
                continue

            agents = [data[0] for data in input_feed]
            indexes = [data[1] for data in input_feed]
            map_tries = [data[2] for data in input_feed]

            for agent in agents:
                agent.car.track = self.tracks[agent.car.track_name]
                agent.car.speed_pre_calc = speed_pre_calc

            alive = len(agents)
            ticks, input_tpa, metal_tpa, tick_tpa = 0, 0, 0, 0

            num_agents = len(agents)
            MetalInstance.init_shaders(num_agents * 10, num_agents * len(points_offset), points_offset, num_agents * 8, num_agents * 4)

            for i, agent in enumerate(agents):
                index = i * 10
                MetalInstance.inVectorBuffer[index:index + 5] = [agent.car.int_x, agent.car.int_y, agent.car.int_direction, track_index[agent.car.track_name], int(agent.car.ppm * 1000)]
            alives = [1] * len(agents)
            corner_lengths = [len(agent.car.future_corners) for agent in agents]
            working[p_id] = 2

            while alive > 0:
                alive = sum([1 for agent in agents if not agent.car.died])
                start_time = time.time()

                for i in range(len(agents)):
                    agent = agents[i]
                    index = i * 10
                    if agent.car.died: continue
                    MetalInstance.inVectorBuffer[index:index + 3] = [agent.car.int_x, agent.car.int_y, agent.car.int_direction]
                    index = i * 8
                    
                    corner_lengths[i] = len(agent.car.future_corners)
                    MetalInstance.cornerBuffer[index:index + 4] = agent.car.future_corners[0]
                    MetalInstance.cornerBuffer[index+4:index+8] = agent.car.future_corners[1]
                        
                tpa_stamp = time.time()

                MetalInstance.pointsAndCorner(len(agents) * len(points_offset), len(agents) * 2)
               
                metal_stamp = time.time()
                per_agents_points = MetalInstance.outVectorBuffer.reshape(len(agents), len(points_offset))
                
                per_agent_corner = MetalInstance.cornerOutBuffer.reshape(len(agents), 4)
                
                for i in range(len(agents)):
                    if alives[i] == 1:
                        if agents[i].car.died:
                            alives[i] = 0
                            MetalInstance.inVectorBuffer[i * 10] = -1
                    if agents[i].car.died: continue
                    agents[i].Tick(ticks, self, per_agents_points[i], per_agent_corner[i])
                    
                tick_stamp = time.time()
                ticks += 1

                metal_tpa += alive * (metal_stamp - tpa_stamp)
                tick_tpa += alive * (tick_stamp - metal_stamp)
                input_tpa += alive * (tpa_stamp - start_time)

                # Log data to log_data Array
                log_data[p_id * 5] = ticks
                log_data[p_id * 5 + 1] = alive
                log_data[p_id * 5 + 2] = int(input_tpa)
                log_data[p_id * 5 + 3] = int(metal_tpa)
                log_data[p_id * 5 + 4] = int(tick_tpa)
                                                                                    
            working[p_id] = 3

            for i in range(len(agents)):
                self.IndividualAgentScore(agents[i], local_lap_times, local_laps, local_scores, max_potentials, map_tries[i], indexes[i])

            with main_lock:
                for i in range(num_agents):
                    scores[i] += local_scores[i]
                    laps[i] += local_laps[i]
                    lap_times[i] += local_lap_times[i]

            local_scores.fill(0)
            local_laps.fill(0)
            local_lap_times.fill(0)

            working[p_id] = 4

    def train_agents_gpu(self):
        if self.generated_starts == []: self.CreateFutureStartsData()

        while self.generating_corners: time.sleep(0.1)
    
        batches = self.CreateAgentsBatches()
        self.agents_feed.extend(batches)
        start = time.time()
        tps_values = [0] * tps_window_size
        prev_ticks = 0

        while any([worker == 0 for worker in self.working[:]]):
            time.sleep(0.1)
            alive_agents, _, tot_ticks, min_ticks, max_ticks, max_alive, min_alive, _, human_formatted, metal_percentage, input_percentage, tick_percentage, TPS, RTS, ts, tc = self.LiveUpdateData(tps_values, start, prev_ticks)
            prev_ticks = tot_ticks
            update_terminal(self, self.map_tries * len(self.environment.agents), alive_agents, tot_ticks, input_percentage, metal_percentage, tick_percentage, TPS, RTS, self.environment.generation, min_ticks, max_ticks, max_alive, min_alive, human_formatted, ts, tc, self.working[:], self.issues.value)
   
        thread = threading.Thread(target=self.CreateFutureStartsData)
        thread.start()
 
        while any([worker != 4 for worker in self.working[:]]):
            time.sleep(update_delay)
            alive_agents, _, tot_ticks, min_ticks, max_ticks, max_alive, min_alive, _, human_formatted, metal_percentage, input_percentage, tick_percentage, TPS, RTS, ts, tc = self.LiveUpdateData(tps_values, start, prev_ticks)
            prev_ticks = tot_ticks
            update_terminal(self, self.map_tries * len(self.environment.agents), alive_agents, tot_ticks, input_percentage, metal_percentage, tick_percentage, TPS, RTS, self.environment.generation, min_ticks, max_ticks, max_alive, min_alive, human_formatted, ts, tc, self.working[:], self.issues.value)

        self.working[:] = [0] * len(self.working)
        self.logger_data = {
            "tps": TPS,
            "rts": RTS,
            "metal_percentage": metal_percentage,
            "input_percentage": input_percentage,
            "tick_percentage": tick_percentage,
            "time_spent": time.time() - start,
            "human_format": human_formatted
        }

        for i in range(len(self.environment.agents)):
            self.environment.agents[i].car.lap_time += self.lap_times[i]
            self.environment.agents[i].car.laps += self.laps[i]
            self.environment.agents[i].car.score += self.scores[i] 
            self.deaths_per_agent[i] = 0
            
        for i in range(self.map_tries): self.max_potentials[i] = 0
            
        self.environment.next_generation(self)
        self.EndOfGeneration()

    def IndividualAgentScore(self, agent, local_lap_times, local_laps, local_scores, max_potentials, map_try, index):
        if agent.car.lap_time > 0:
            local_scores[index] += 1 * score_multiplier
            local_laps[index] += 1
            local_lap_times[index] += agent.car.lap_time
        elif agent.car.lap_time == -1:
            return
        else:
            if max_potentials[map_try] == 0:
                max_potentials[map_try] = agent.car.CalculateMaxPotential()
            agent_seen = agent.car.ScoreCar(self.track_scores[agent.car.track_name], self)
            score = agent_seen / max_potentials[map_try]
            
            if score >= 1: 
                agent.car.CalculateMaxPotential()
                if abs(agent.car.seen - agent.car.max_pot_seen) < 40: return
                self.issues.value += 1
                print(f"Weird score..., track: {agent.car.track_name}, start_x-start_y: {agent.car.start_x}-{agent.car.start_y}, start_dir: {agent.car.start_direction}, final_x-final_y: {agent.car.finish_x}-{agent.car.finish_y}, seen: {agent.car.seen}, max_pot: {agent.car.max_pot_seen}")
                # Weird score, shouldnt be saved
                return
            local_scores[index] += score * score_multiplier

    def EndOfGeneration(self):
        print("ended generation")
        self.environment.previous_completion.insert(0, self.environment.previous_best_score)
        self.environment.previous_lap_times.insert(0, self.environment.previous_best_lap)

        for i in range(len(self.environment.agents)):
            self.scores[i] = 0
            self.lap_times[i] = 0
            self.laps[i] = 0

        self.CalculateProgressionValues()

    def LiveUpdateData(self, tps_values, start, prev_ticks):
        alive_agents, ticks, min_ticks, max_ticks, max_alive, min_alive, tot_input, tot_metal, tot_tick = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i in range(int(len(self.log_data)/5)):
            ticks += self.log_data[i*5]
            alive_agents += self.log_data[i*5 + 1]
            tot_input += self.log_data[i*5 + 2]
            tot_metal += self.log_data[i*5 + 3]
            tot_tick += self.log_data[i*5 + 4]
            if min_ticks == 0 or min_ticks > self.log_data[i*5]: min_ticks = self.log_data[i*5]
            if max_alive < self.log_data[i*5 + 1]: max_alive = self.log_data[i*5 + 1]
            if min_alive == 0 or min_alive > self.log_data[i*5 + 1]: min_alive = self.log_data[i*5 + 1]
            if max_ticks < self.log_data[i*5]: max_ticks = self.log_data[i*5]
        tot_ticks = ticks
        ticks /= len(self.log_data)/5
        time_spent = time.time() - start
        human_formatted = time.strftime("%H:%M:%S", time.gmtime(time_spent))[3:]
        metal_percentage = round(tot_metal / (tot_input + tot_metal + tot_tick + 1) * 100, 1)
        input_percentage = round(tot_input / (tot_input + tot_metal + tot_tick + 1) * 100, 1)
        tick_percentage = round(tot_tick / (tot_input + tot_metal + tot_tick + 1) * 100, 1)
        TPS = int(tot_ticks / (time_spent))
        RTS = round(TPS * delta_t, 1)
        if any([True for worker in self.working[:] if worker == 2]):
            tps = round((tot_ticks - prev_ticks) / update_delay, 2) * self.options['cores'] / (sum([1 for worker in self.working[:] if worker == 2]) + 0.0001)
            tps_values.pop(0)
            tps_values.append(tps)
            smoothed_tps = round(sum(tps_values) / len(tps_values), 1)
        else:
            smoothed_tps = round(sum(tps_values) / len(tps_values), 1)
        
        return alive_agents, ticks, tot_ticks, min_ticks, max_ticks, max_alive, min_alive, smoothed_tps, human_formatted, metal_percentage, input_percentage, tick_percentage, TPS, RTS, self.trajectory_score, self.trajectory_completion
    
    def CalculateProgressionValues(self):
        if len(self.environment.previous_lap_times) > 1: 
            weights = [np.exp(-i/30) for i in range(len(self.environment.previous_lap_times)-1)]
            trajectory_completion = round(sum([(self.environment.previous_completion[i] - self.environment.previous_completion[i+1]) * weights[i] for i in range(len(self.environment.previous_completion) - 2)]) / sum(weights), 2)
        else:
            trajectory_completion = 0

        non_zero_lap_times = [i for i in self.environment.previous_lap_times if i != 0]
        if len(non_zero_lap_times) > 1:
            trajectory_score = round(sum([(non_zero_lap_times[i] - non_zero_lap_times[i+1]) * weights[i] for i in range(len(non_zero_lap_times) - 2)]) / sum(weights), 2)
        else:
            trajectory_score = 0
        self.trajectory_score = trajectory_score
        self.trajectory_completion = trajectory_completion

    def CreateAgentsBatches(self):
        all_agents = []
        all_corners, starts, start_tracks = self.generated_corners, self.generated_starts, self.generated_start_tracks

        for i in range(self.map_tries):
            for k in range(len(self.environment.agents)):
                new_agent = Agent(self.options['environment'], self.tracks[start_tracks[i]], starts[i][0], starts[i][1], start_tracks[i], False)
                new_agent.network = self.environment.agents[k].network
                new_agent.SetAgent(starts[i], self.tracks[start_tracks[i]], start_tracks[i])
                
                if self.player == 3: 
                    new_agent.car.acceleration = 1
                    new_agent.car.speed = self.config['quali_start_speed'].get(start_tracks[i])
               
                new_agent.car.future_corners = np.array(all_corners[i], copy=True).tolist()
                if self.player in [1, 2]:
                    new_agent.car.direction += (random.random() - 0.5) * 5
                    new_agent.car.start_direction = new_agent.car.direction

                new_agent.car.track = []
                new_agent.mutation_strengh = self.mutation_strength
                all_agents.append([new_agent, k, i])
        random.shuffle(all_agents)
       
        batches = [[] for _ in range(self.options['cores'])]
        
        for i, agent in enumerate(all_agents):
            agent, real_agent_index, map_try_num = agent
            batches[i % self.options['cores']].append([agent, real_agent_index, map_try_num])

        return batches
    
    def load_single_track(self):
        self.tracks = {}
        self.start_positions = {}
        self.real_starts = {}
        self.corners = {}
        self.track_scores = {}
        track_name = self.options['track_name']
        data = self.load_track(track_name)

        self.tracks[track_name] = np.array(data['track'])
        self.track = self.tracks[track_name]
        self.start_positions[track_name] = data['start_positions']
        self.real_starts[track_name] = data['real_start']
        self.corners[track_name] = data['corners']
        self.track_scores[track_name] = data['scores']

        self.track = random.choice(list(self.tracks.values()))
        self.track_name = [name for name, track in self.tracks.items() if track is self.track][0]

        if self.player in [0, 3, 4, 6, 7]:
            self.track_name = self.options['track_name']
            self.track = self.tracks[self.track_name]

        self.track_names = list(self.tracks.keys())
        self.start_pos = [0, 0]

        if random_start_position and self.player not in [3, 4, 6, 7, 8]:
            self.start_pos, self.start_dir = random.choice(self.start_positions[self.track_name])
        else:
            self.start_pos[0] = self.real_starts[self.track_name][0][0]
            self.start_pos[1] = self.real_starts[self.track_name][0][1]
            self.start_dir = self.real_starts[self.track_name][1]

        if self.player == 3: self.map_tries = 1

        print(f" * Loaded track: {track_name}")

    def CreateFutureStartsData(self):
        self.generating_corners = True
        starts, start_tracks = self.GenerateTrainingStarts()
        all_corners = [[] for _ in range(self.map_tries)]
        for i in range(self.map_tries):
            new_agent = Agent(self.options['environment'], self.tracks[start_tracks[i]], starts[i][0], starts[i][1], start_tracks[i])
            new_agent.SetAgent(starts[i], self.tracks[start_tracks[i]], start_tracks[i])
            new_agent.car.UpdateCorners()
            new_agent.car.setFutureCorners(self.corners[start_tracks[i]])
            all_corners[i] = new_agent.car.future_corners
        
        self.generated_corners = all_corners
        self.generated_starts = starts
        self.generated_start_tracks = start_tracks

        self.generating_corners = False


    def GenerateTrainingStarts(self):
        start_tracks, starts, chosen_tracks = [], [], []

        if self.player == 3:
            start_tracks = [self.track_name]
            start_x = self.real_starts[self.track_name][0][0]
            start_y = self.real_starts[self.track_name][0][1]
            starts.append([[start_x, start_y], self.real_starts[self.track_name][1]])
            return starts, start_tracks
    
        count = 0
        while len(chosen_tracks) < real_starts_num and count < 10000:
            track = random.choice(list(self.tracks.keys()))
            if track not in chosen_tracks:
                start_tracks.append(track)
                start_x = self.real_starts[track][0][0]
                start_y = self.real_starts[track][0][1]
                starts.append([[start_x, start_y], self.real_starts[track][1]])
                chosen_tracks.append(track)
            count += 1
        self.real_start_tracks = chosen_tracks
        
        while len(starts) < self.map_tries:
            track = random.choice(list(self.tracks.keys()))
            start_tracks.append(track)
            rand_start = random.choice(self.start_positions[track])
            starts.append(rand_start)
            
        for i in range(len(start_tracks)):
            assert(self.tracks[start_tracks[i]][starts[i][0][0], starts[i][0][1]] == 10)

        return starts, start_tracks

    def load_all_tracks(self):
        self.start_positions = {}
        self.tracks = {}
        if self.shared_tracks is None:
            self.shared_tracks = Manager().dict()
        self.real_starts = {}
        self.corners = {}
        self.track_scores = {}
        for file in os.listdir("./data/tracks"):
            if file.endswith(".png") and not file.endswith("_surface.png") and not file.endswith("_path.png"):
                print(f" - Loading track: {file[:-4]}         \r", end='', flush=True)

                if self.player == 8 and file[:-4] != self.options['track_name']: continue
                self.track_name = file[:-4]
                data = self.load_track(file[:-4])
                
                self.tracks[file[:-4]] = np.array(data['track']).astype(np.int8)
                if self.player in [1,2,3]:
                    self.shared_tracks[file[:-4]] = self.tracks[file[:-4]]
                self.track = self.tracks[file[:-4]]

                self.corners[file[:-4]] = data['corners']
                self.start_positions[file[:-4]] = data['start_positions']
                self.real_starts[file[:-4]] = data['real_start']
                self.track_scores[file[:-4]] = data['scores']
        
        print(f" * Loaded tracks: {str(', '.join(list(self.tracks.keys())))}")
        self.track_name = random.choice(list(self.tracks.keys()))
        self.track = self.tracks[self.track_name]

        if self.player in [0, 6, 7]:
            self.track_name = self.options['track_name']
            self.track = self.tracks[self.track_name]

        self.track_names = list(self.tracks.keys())
        self.start_pos = [0, 0]
        if random_start_position and self.player not in [3, 4, 6, 7, 8]:
            self.start_pos, self.start_dir = random.choice(self.start_positions[self.track_name])
        else:
            self.start_pos[0] = self.real_starts[self.track_name][0][0]
            self.start_pos[1] = self.real_starts[self.track_name][0][1]
            self.start_dir = self.real_starts[self.track_name][1]
        
        self.agent = False

        if self.player in [4, 8]:
            self.tracks = {}
            self.tracks[self.track_name] = self.track
            self.track_name = self.options['track_name']

    def load_track(self, track_name):
        if os.path.exists("./data/tracks/" + track_name + ".npy"):
            track = np.load("./data/tracks/" + track_name + '.npy', allow_pickle=True).item()
            return track
        elif os.path.exists("./data/tracks/" + track_name + ".png"):
            print(f" - Generating track: {track_name}")
            image = Image.open("./data/tracks/" + track_name + ".png") 
            image = image.convert("RGB")
            width, height = image.size

            self.board = [[0] * width for _ in range(height)]
            real_start = None
            for y in range(height):
                for x in range(width):
                    pixel_value = image.getpixel((x, y))
                    if pixel_value == (0, 0, 0):
                        self.board[y][x] = 1
                    elif is_color_within_margin(pixel_value, (255, 147, 0), 20):
                        self.board[y][x] = 2
                    elif is_color_within_margin(pixel_value, (255, 0, 0), 50):
                        self.board[y][x] = 3
                    elif is_color_within_margin(pixel_value, (34, 255, 6), 5):
                        real_start = [[x, y], self.config['start_dir'].get(track_name)]
                        self.board[y][x] = 1

            self.track = np.array(self.board)
            
            print(" - Finding center line")
            self.find_center_line()
            center_line_coords = np.argwhere(self.track == 10).tolist()
            with open('./src/config.json', 'r') as json_file:
                config_data = json.load(json_file)
            real_length = config_data['real_track_lengths'].get(track_name)
            ppm = round(len(center_line_coords) / real_length, 3)
            # Edit config.json to add/edit the ppm
            with open('./src/config.json', 'w') as json_file:
                config_data['pixel_per_meter'][track_name] = ppm
                json.dump(config_data, json_file, indent=4)
            real_start_x = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[0]
            real_start_y = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[1]
            real_start = [[real_start_y, real_start_x], real_start[1]]
            print(" - Generating center line inputs")
            corners, track_intensity= self.setup_corners()
            assert(real_start != None)
            
            print(" - Generating new starts")
            data = {
                "track": self.track,
                "start_positions": get_new_starts(self.track, 5000, track_intensity, track_name),
                "corners": corners,
                "real_start": real_start,
                "scores": {}
            }
            
            np.save("./data/tracks/" + track_name, data)
            print(" - Smoothening track")
            smoothen(track_name)

            print(f" - Generated track: {track_name}")
            if self.player == 0: self.exit() # Useful when generating all tracks
            else: return self.load_track(track_name)
        else: self.exit("Not able to find track with this name")
    def setup_corners(self):
        corners, _, data = get_corners(self.track, self.track_name, self.config.get("thresholds").get(self.track_name))
        treated_corners = []
        for corner in corners:
            treated_corners.append([(corner[0], corner[1]), data[corner[1], corner[0]]])
        return treated_corners, data

    def find_center_line(self):
        binary_image = (self.track != 0).astype(np.uint8)

        dist_transform = distance_transform_edt(binary_image)

        threshold = 0.2 * dist_transform.max()
        centerline_binary = (dist_transform > threshold) * 255

        matrix_with_centerline = self.track.copy()
        matrix_with_centerline[centerline_binary == 255] = 10

        skeleton = skeletonize(centerline_binary)

        thinned_centerline = skeleton * 10

        for i in range(len(thinned_centerline)):
            for j in range(len(thinned_centerline[0])):
                if thinned_centerline[i][j] == 10:
                    self.track[i][j] = 10
    
    def load_best_agent(self, path):
        best_agent = 0
        for file in os.listdir(path):
            if file.endswith(".npy") and not file.endswith("gents.npy"):
                if int(file[11:-4]) > best_agent:
                    best_agent = int(file[11:-4])
        print(f" - Loading best agent: {best_agent}")
        agent = Agent(self.options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
        agent.network = np.load(f"{path}/best_agent_" + str(best_agent) + ".npy", allow_pickle=True).item()['network']
        return best_agent, agent

    def extract_csv(self, path):
        try:
            with open(path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                self.environment.previous_best_score = float(last_line.split(",")[1])
                self.environment.previous_best_lap = float(last_line.split(",")[2])
        except:
            pass
    def performanceTest(self):
        tick_time = 0
        self.debug = True
        
        for track_name in self.track_names:
            track = self.tracks[track_name]
            pos, direction = random.choice(self.start_positions[track_name])
            corners = [(np.random.uniform(0, 5000), np.random.uniform(0, 500), np.random.uniform(0, 360), np.random.uniform(0, 90)) for _ in range(len(self.corners[track_name]))]
            corners = [(int(c1), int(c2), int(c3), int(c4)) for (c1, c2, c3, c4) in corners]
            for agent in self.environment.agents:
                agent.car.track = track
                agent.car.track_name = track_name
                agent.car.x = pos[1]
                agent.car.y = pos[0]
                agent.car.direction = direction
                agent.car.UpdateCorners()
                agent.car.future_corners = corners
            
            rand = np.array([np.random.uniform(0, 200)] * len(points_offset))
            rand_4 = np.array([np.random.uniform(0, 200)] * 4)
            start_time = time.time()
            for agent in self.environment.agents:
                for i in range(perft_ticks):
                    agent.Tick(i, self, rand, rand_4)
            tick_time += time.time() - start_time
        score = tick_time / len(self.track_names) / len(self.environment.agents) / perft_ticks * 1000
        self.totalScore += score
        self.runs += 1
        self.environment = Environment(self.options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name, )

        for i in range(len(self.environment.agents)):
            self.environment.agents[i].car.speed_pre_calc = self.pre_calc_speed
    
    def exit(self, message="Exiting..."):
        print(message)
        self.running.value = False
        import pygame
        import sys
        pygame.quit()
        sys.exit()

    def multiple_players_shader_init(self):
        for i, agent in enumerate(self.environment.agents):
            index = i * 10
            self.Metal.inVectorBuffer[index:index + 5] = [agent.car.int_x, agent.car.int_y, agent.car.int_direction, self.track_index[agent.car.track_name], int(agent.car.ppm * 1000)]

    

