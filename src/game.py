import gc
import os
import numpy as np
import random
import time
import json

from multiprocessing import Process, Array, Manager, Value, RawArray

from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None

from environment import Environment
from car import Car
from utils import is_color_within_margin, calculate_distance, get_new_starts, get_nearest_centerline
from agent import Agent
from settings import *
from smoothen import smoothen
from metal import Metal
from corners import get_corners

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

        self.retries = []

        self.shared_tracks = None

        with open("./src/config.json") as f:
            self.config = json.load(f)

        if self.player in [0, 3, 4, 7, 9]:
            self.load_single_track()
        else:
            self.load_all_tracks()

        self.track_results = {}
        if self.player in [1, 2]:
            for track in self.track_names:
                self.track_results[track] = 0
        
        self.environment = Environment(game_options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)
        
        if self.player == 0:
            self.car = Car(self.track, self.start_pos, self.start_dir, self.track_name)
            self.car.setFutureCorners(self.corners[self.track_name])
        elif self.player == 2:
            self.environment.load_agents(self)
            self.player = 1
        elif self.player == 3:
            self.environment.load_specific_agents(self)
        elif self.player == 4:
            best_agent, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained")
            self.extract_csv(f"./data/per_track/{self.track_name}/log.csv")
            agent.car.track = self.track
            agent.car.track_name = self.track_name
            self.environment.generation = best_agent
            self.environment.agents[0] = agent
            
            self.environment.agents[0].car.speed = self.config['quali_start_speed'].get(self.track_name)
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
            _, agent = self.load_best_agent("./data/train/trained")

            self.environment.agents[1] = agent
            self.environment.agents[0] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
        elif self.player == 7:
            if game_options['generation_to_load'] == 0:
                best_agent, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained/")
            else:
                best_agent = game_options['generation_to_load']
                agent = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                agent.network = np.load("./data/per_track/" + self.track_name + "/trained/best_agent_" + str(best_agent) + ".npy", allow_pickle=True).item()['network']
            agent.car.speed = self.config['quali_start_speed'].get(self.track_name)
            start_x = self.real_starts[self.track_name][0][1]
            start_y = self.real_starts[self.track_name][0][0]
            agent.car.x = start_x
            agent.car.start_x = start_x
            agent.car.y = start_y
            agent.car.start_y = start_y
            agent.car.direction = self.real_starts[self.track_name][1]
            agent.car.start_direction = self.real_starts[self.track_name][1]

            self.best_agent = best_agent
            self.environment.agents[0] = agent
            
        elif self.player == 8:
            networks = np.load("./data/train/agents.npy", allow_pickle=True).item()['networks']
            self.environment.agents = [None] * game_options['environment']['num_agents']
            for i in range(len(self.environment.agents)):
                self.environment.agents[i] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                self.environment.agents[i].network = networks[i]
                self.environment.agents[i].car.speed = 0
            self.environment.agents = self.environment.agents[::-1]
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

        if self.player in [0, 4, 5, 7, 8]:
            self.Metal = Metal(self.tracks)
            self.track_index = self.Metal.getTrackIndexes()
            self.Metal.init_shaders(len(points_offset) * 5 * len(self.environment.agents), len(points_offset) * len(self.environment.agents), points_offset)

        self.prev_update = time.time()
        self.start_time = time.time()

    def tick(self):
        if self.player == 0:
            self.ticks += 1
            if self.ticks % 100 == 0 and debug:
                print("Ticks: " + str(self.ticks))
            self.car.Tick(self)
            if self.car.died == True:
                self.restart = True
                self.car.died = False
                if self.debug: print("Car died, restarting")
        elif self.player == 1 or self.player == 3:
            self.train_agents_gpu()
        elif self.player == 4:
            self.environment.agents[0].Tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 5:
            import pygame
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_h]:
                self.environment.agents[0].car.ApplyPlayerInputs()
                self.environment.agents[0].car.UpdateCar()
                self.environment.agents[0].car.CheckCollisions(self.ticks)
            else:
                self.environment.agents[0].Tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 7:
            self.environment.agents[0].Tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 8:
            input_data = []
            for agent in self.environment.agents:
                for offset in points_offset:
                    input_data += [int(agent.car.x), int(agent.car.y), int(agent.car.direction + offset + 90) % 360, self.track_index[agent.car.track_name], int(agent.car.ppm * 1000)]
            out_data = self.Metal.getPointsOffset(np.array(input_data).flatten().astype(np.int32))
            per_agents_points = out_data.reshape((len(self.environment.agents), len(points_offset)))
            for i in range(len(self.environment.agents)):
                self.environment.agents[i].Tick(self.ticks, self, per_agents_points[i])
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

        self.agents_feed = Manager().list()
        self.main_lock = Manager().Lock()
        self.lap_times = Array('i', [0] * num_agents, lock=False)
        self.laps = Array('i', [0] * num_agents, lock=False)
        self.scores = Array('d', [0] * num_agents, lock=False)
        self.working = Array('b', [False] * num_agents, lock=False)
        self.log_data = Array('i', [0] * num_processes * 5, lock=False)

        for i in range(num_processes):
            process = Process(target=self.create_process, args=(self.agents_feed, self.log_data, self.scores, self.laps, self.lap_times, self.working, i, self.main_lock))
            processes.append(process)
        
        for i in range(len(processes)):
            print(f" - Starting process {i+1}/{len(processes)}         \r", end='', flush=True)
            processes[i].start()
        print(f" * Started {len(processes)} processes, running {len(self.environment.agents) * self.map_tries} agents in total, on {len(self.track_names)} tracks")
    
    def create_process(self, agents_feed, log_data, scores, laps, lap_times, working, p_id, main_lock):
        local_scores = [0] * len(self.environment.agents)
        local_laps = [0] * len(self.environment.agents)
        local_lap_times = [0] * len(self.environment.agents)

        MetalInstance = Metal(self.tracks)
        track_index = MetalInstance.getTrackIndexes()
        self.tracks = []
        for agent in self.environment.agents:
            agent.car.track = []

        while True:
            try:
                input_feed = agents_feed.pop(0)
                working[p_id] = True
            except:
                time.sleep(0.5)
                continue

            agents = [data[0] for data in input_feed]
            indexes = [data[1] for data in input_feed]

            for agent in agents:
                agent.car.lap_time = 0
                agent.car.score = 0
                agent.car.laps = 0

            alive = len(agents)
            ticks = 0
            input_tpa = 0
            metal_tpa = 0
            tick_tpa = 0

            MetalInstance.init_shaders(len(agents) * 5, len(agents) * len(points_offset), points_offset)

            while alive > 0:
                alive = sum([1 for agent in agents if not agent.car.died])
                start_time = time.time()
                for i in range(len(agents)):
                    agent = agents[i]
                    index = i * 5
                    MetalInstance.inVectorBuffer[index:index + 5] = [int(agent.car.x), int(agent.car.y), int(agent.car.direction), track_index[agent.car.track_name], int(agent.car.ppm) * 1000]
                    
                tpa_stamp = time.time()
                MetalInstance.getPointsOffset(len(agents) * len(points_offset))
               
                metal_stamp = time.time()
                per_agents_points = MetalInstance.outVectorBuffer.reshape((len(agents), len(points_offset)))
                for i in range(len(agents)):
                    agents[i].Tick(ticks, self, per_agents_points[i])
                tick_stamp = time.time()
                del per_agents_points
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
            
            max_potentials = {}
            for track in self.track_names:
                max_potentials[track] = 0
            for i in range(len(agents)):
                if agents[i].car.lap_time > 0:
                    local_scores[indexes[i]] += 1 * score_multiplier
                    local_laps[indexes[i]] += 1
                    local_lap_times[indexes[i]] += agents[i].car.lap_time
                else:
                    if max_potentials[agents[i].car.track_name] == 0:
                        max_potentials[agents[i].car.track_name] = agents[i].car.CalculateMaxPotential()
                    score = agents[i].car.CalculateScore(max_potentials[agents[i].car.track_name])
                    local_scores[indexes[i]] += score * score_multiplier
                    if score == 1: # Very weird, shouldnt happen (Edge case?). Seemed to fix a rare issue
                        local_laps[indexes[i]] += 1
                        local_lap_times[indexes[i]] += agents[i].car.lap_time

            with main_lock:
                for i in range(len(self.environment.agents)):
                    scores[i] += local_scores[i]
                    laps[i] += local_laps[i]
                    lap_times[i] += local_lap_times[i]
                    local_scores[i] = 0
                    local_laps[i] = 0
                    local_lap_times[i] = 0
            working[p_id] = False
            gc.collect()

    def train_agents_gpu(self):
        starts, start_tracks = self.GenerateTrainingStarts()

        all_agents = []

        for i in range(self.map_tries):
            all_corners = []
            for k in range(len(self.environment.agents)):
                print(f" - Creating agents | {i * len(self.environment.agents) + k + 1}/{self.map_tries * len(self.environment.agents)}        \r", end='', flush=True)
                agent = self.environment.agents[k]
                start_track = start_tracks[i]
                new_agent = Agent(self.options['environment'], self.tracks[start_track], starts[i][0], starts[i][1], start_track)
                new_agent.network = agent.network
                new_agent.SetAgent(starts[i], self.tracks[start_track], start_track)
                new_agent.car.UpdateCorners()
                
                if self.player == 3: new_agent.car.speed = self.config['quali_start_speed'].get(start_track)
                if k == 0:
                    new_agent.car.setFutureCorners(self.corners[start_track])
                    all_corners = new_agent.car.future_corners
                else:
                    new_agent.car.future_corners = np.array(all_corners, copy=True).tolist()
                all_agents.append([new_agent, k])
     
        random.shuffle(all_agents)

        self.batches = [[] for _ in range(self.options['cores'])]
        
        for i, agent in enumerate(all_agents):
            agent, real_agent_index = agent
            self.batches[i % self.options['cores']].append([agent, real_agent_index])

        self.agents_feed.extend(self.batches)

        while sum(self.working[:]) < self.options['cores']:
            time.sleep(0.1)
            print(f" - Waiting for agents to start | Started: {sum(self.working[:])}, Left: {len(self.agents_feed)}         \r", end='', flush=True)
        
        start = time.time()

        prev_ticks = 0
        update_delay = 0.1
        tps_window_size = 20  # Adjust the window size as needed
        tps_values = [0] * tps_window_size

        while any(self.working):
            time.sleep(update_delay)
            alive_agents, ticks, min_ticks, max_ticks, max_alive, min_alive, tot_input, tot_metal, tot_tick = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for i in range(int(len(self.log_data[:])/5)):
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
            if sum(self.working[:]) != 0:
                tps = round((tot_ticks - prev_ticks) / update_delay, 2) * self.options['cores'] / sum(self.working[:])

                # Update the moving average of TPS
                tps_values.pop(0)
                tps_values.append(tps)
                smoothed_tps = round(sum(tps_values) / len(tps_values), 1)

                prev_ticks = tot_ticks
            else:
                smoothed_tps = round(sum(tps_values) / len(tps_values), 1)
        
            print(f" - Agents Training | Alive: Min/Max/Avg {min_alive}/{max_alive}/{int(alive_agents/(len(self.log_data)/5))}, Ticks: Min/Max/Avg {min_ticks}/{max_ticks}/{int(ticks)}, Input/Metal/Tick: {input_percentage}/{metal_percentage}/{tick_percentage}%, TPS: {smoothed_tps}  | {human_formatted}        \r", end='', flush=True)

        for i in range(len(self.environment.agents)):
            self.environment.agents[i].car.lap_time += self.lap_times[i]
            self.environment.agents[i].car.laps += self.laps[i]
            self.environment.agents[i].car.score += self.scores[i] 
            
        self.environment.next_generation(self)

        print(f" - Generation: {self.environment.generation - 1}, completion: {(self.environment.previous_best_score/ (score_multiplier * self.map_tries) * 100):0.2f}%, laps: {max(self.laps)}, lap time: {self.environment.previous_best_lap}                    ")
        
        for i in range(len(self.environment.agents)):
            self.scores[i] = 0
            self.lap_times[i] = 0
            self.laps[i] = 0

        gc.collect()
    
    def load_single_track(self):
        self.tracks = {}
        self.start_positions = {}
        self.real_starts = {}
        self.corners = {}
        track_name = self.options['track_name']
        data = self.load_track(track_name)

        self.tracks[track_name] = np.array(data['track'])
        self.track = self.tracks[track_name]
        self.start_positions[track_name] = data['start_positions']
        self.real_starts[track_name] = data['real_start']
        self.corners[track_name] = data['corners']

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

    def GenerateTrainingStarts(self):
        start_tracks, starts, weights, tracks, chosen_tracks = [], [], [], [], []

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

    def load_all_tracks(self, shared=False):
        self.start_positions = {}
        self.tracks = {}
        if self.shared_tracks is None:
            self.shared_tracks = Manager().dict()
        self.real_starts = {}
        self.corners = {}
        for file in os.listdir("./data/tracks"):
            if file.endswith(".png") and not file.endswith("_surface.png"):
                print(f" - Loading track: {file[:-4]}         \r", end='', flush=True)
                folder_path = f"./data/per_track/{file[:-4]}/trained"
                os.makedirs(folder_path, exist_ok=True)

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
            start_positions = []

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
                        exists = False
                        self.board[y][x] = 1
                        for pos in start_positions:        
                            if calculate_distance(pos, (x, y)) < 10:
                                exists = True
                        if not exists:
                            start_positions.append((x, y))
                            self.board[y][x] = 20 + len(start_positions)
                            if len(start_positions) == 1:
                                real_start = [[x, y], self.config['start_dir'].get(track_name)]
            self.track=np.array(self.board)
            
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

            print(" - Generating new starts")
            starts = get_new_starts(self.track, 1000)
            real_start_x = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[0]
            real_start_y = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[1]
            real_start = [[real_start_y, real_start_x], real_start[1]]
            print(" - Generating center line inputs")
            assert(real_start != None)
            corners, _, _ = get_corners(self.track)
            data = {
                "track": self.track,
                "start_positions": starts,
                "corners": corners,
                "real_start": real_start
            }
            
            np.save("./data/tracks/" + track_name, data)
            print(" - Smoothening track")
            smoothen(track_name)

            print(f" - Generated track: {track_name}")
            if self.player == 0:
                self.running.value = False
                import pygame
                import sys
                pygame.quit()
                sys.exit()
            else:
                return self.load_track(track_name)
        else:
            print("Not able to find track with this name")
            import pygame
            import sys
            pygame.quit()
            sys.exit()

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
            for agent in self.environment.agents:
                agent.car.track = track
                agent.car.track_name = track_name
                agent.car.x = pos[1]
                agent.car.y = pos[0]
                agent.car.direction = direction
                agent.car.UpdateCorners()
            start_time = time.time()
            for agent in self.environment.agents:
                for i in range(perft_ticks):
                    agent.Tick(i, self, np.array([np.random.uniform(0, 200)] * len(points_offset)))
            tick_time += time.time() - start_time
        score = tick_time / len(self.track_names) / len(self.environment.agents) / perft_ticks * 1000
        self.totalScore += score
        self.runs += 1
        self.environment = Environment(self.options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)