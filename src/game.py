import os
import numpy as np
import random
import threading
import time

from multiprocessing import Process, Array, Manager, Value

from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None

from environment import Environment
from car import Car
from utils import is_color_within_margin, calculate_distance, get_new_starts, print_progress, get_nearest_centerline
from agent import Agent
from settings import *
from precomputed import start_dir, offsets, directions, real_track_lengths
from smoothen import main as smoothen

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

        if self.player in [0, 3, 4, 7, 9]:
            self.load_single_track()
        else:
            self.load_all_tracks()
            
        self.environment = Environment(game_options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)

        if self.player == 0:
            self.car = Car(self.track, self.start_pos, self.start_dir, self.track_name)
        elif self.player == 2:
            self.environment.load_agents()
            self.player = 1
        elif self.player == 3:
            self.environment.load_specific_agents(self)
        elif self.player == 4:
            best_agent, agent = self.load_best_agent(f"./data/per_track/{self.track_name}/trained")
            self.extract_csv(f"./data/per_track/{self.track_name}/log.csv")
            self.environment.generation = best_agent
            self.environment.agents[0] = agent

            self.environment.agents[0].car.x = self.real_starts[self.track_name][0][1]
            self.environment.agents[0].car.y = self.real_starts[self.track_name][0][0]
            self.environment.agents[0].car.direction = start_dir[self.track_name]
        elif self.player == 5:
            best_agent, agent = self.load_best_agent("./data/train/trained")
            self.extract_csv("./data/train/log.csv")

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
            agent.car.speed = 0
            start_x = self.real_starts[self.track_name][0][1]
            start_y = self.real_starts[self.track_name][0][0]
            agent.car.x = start_x
            agent.car.y = start_y
            agent.car.direction = start_dir[self.track_name]

            self.best_agent = best_agent
            self.environment.agents[0] = agent
            
        elif self.player == 8:
            self.environment.agents = []
            # - Go from max to lowest, and add num_agents of the best agents with a different score
            possible_agent_number = [] 
            possible_agent_laps = []
            # - Obtain all agent numbers with distinct lap times
            with open("./data/per_track/" + self.track_name + "/log.csv", "r") as f:
                lines = f.readlines()[1:]
                for line in lines:
                    if line.split(",")[2] not in possible_agent_laps:
                        possible_agent_laps.append(line.split(",")[2])
                        possible_agent_number.append(line.split(",")[0])
            possible_agent_number = possible_agent_number[-game_options['environment']['num_agents']:]
            self.environment.agents = [None] * game_options['environment']['num_agents']
            for i in range(len(possible_agent_number)):
                self.environment.agents[i] = Agent(game_options['environment'], self.track, self.start_pos, self.start_dir, self.track_name)
                self.environment.agents[i].network = np.load("./data/per_track/" + self.track_name + "/trained/best_agent_" + str(possible_agent_number[i]) + ".npy", allow_pickle=True).item()['network']
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
                self.environment.agents[i].car.speed = 0
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
                agent.car.direction = start_dir[self.track_name]
                assert(agent.car.track[agent.car.y, agent.car.x] == 10)
        elif self.player == 10:
            print(" - Starting performance test...")
            self.totalScore = 0
            self.runs = 0
            self.prev_update = time.time()
            self.start_time = time.time()
        elif self.player == 11:
            data = np.load("./data/per_track/" + self.track_name + "/generated_run.npy", allow_pickle=True).item()
            self.environment.agents = data['agents']
            self.generated_data = data['info']
            for agent in self.environment.agents:
                agent.car.x = self.real_starts[self.track_name][0][1]
                agent.car.y = self.real_starts[self.track_name][0][0]
                agent.car.direction = start_dir[self.track_name]
                agent.car.track = self.track
                assert(agent.car.track[agent.car.y, agent.car.x] == 10)

        if self.player in [1, 2, 3]:
            self.initialise_process()

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
            self.train_agents()
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
            for agent in self.environment.agents:
                agent.Tick(self.ticks, self)
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
        self.waiting_for_agents = Value('b', False)
        self.main_lock = Manager().Lock()
        self.lap_times = Array('d', [0] * num_agents)
        self.laps = Array('i', [0] * num_agents)
        self.scores = Array('d', [0] * num_agents)
        self.results = Array('b', [0] * len(self.environment.agents) * self.map_tries)
        self.working = Array('b', [False] * num_processes)
        self.secondary_lock = Manager().Lock()

        for i in range(num_processes):
            process = Process(target=self.create_process, args=(self.agents_feed, self.waiting_for_agents, self.main_lock, self.lap_times, self.laps, self.scores, self.running, self.working, i, self.results))
            processes.append(process)
        
        for i in range(len(processes)):
            print(f" - Starting process {i+1}/{len(processes)}         \r", end='', flush=True)
            processes[i].start()
        print(f" - Started {len(processes)} processes")

    def create_process(self, agents_feed, waiting_for_agents, main_lock, lap_times, laps, scores, running, working, p_id, results):
        local_lap_times = [0] * len(self.environment.agents)
        local_laps = [0] * len(self.environment.agents)
        local_scores = [0] * len(self.environment.agents)
        updated = False
        while running.value:
            try:
                if waiting_for_agents.value == True: raise Exception("Waiting for agents")
                agent_index, try_num, agent, start_track, start = agents_feed.pop()
            except:
                if updated == True:
                    with main_lock:
                        for i in range(len(local_scores)):
                            lap_times[i] += local_lap_times[i]
                            laps[i] += local_laps[i]
                            scores[i] += local_scores[i]
                    local_lap_times = [0] * len(self.environment.agents)
                    local_laps = [0] * len(self.environment.agents)
                    local_scores = [0] * len(self.environment.agents)
                    updated = False
                    working[p_id] = False
                time.sleep(0.1)
                continue
            if updated == False: working[p_id] = True
            if agent.car.died: agent.car.died = False
            agent.car.x = start[0][1]
            agent.car.y = start[0][0]
            agent.car.direction = start[1]
            agent.car.start_x = start[0][1]
            agent.car.start_y = start[0][0]
            agent.car.start_dir = start[1]
            agent.car.track = self.tracks[start_track]
            agent.car.track_name = start_track
            agent.car.lap_time = 0
            ticks = 0
            while not agent.car.died:
                agent.Tick(ticks, self)
                ticks += 1
            if agent.car.lap_time != 0:
                local_lap_times[agent_index] += agent.car.lap_time
                local_laps[agent_index] += 1
            local_scores[agent_index] += min(1 * score_multiplier, int((agent.car.CalculateScore() * score_multiplier)))
            updated = True
            
    def train_agents(self):

        starts, start_tracks = self.GenerateTrainingStarts()
        
        self.waiting_for_agents.value = True
        data_to_append = []
        for i in range(len(self.environment.agents)):
            for j in range(self.map_tries):
                print(f" - Feeding agents to processes, {i*j}/{len(self.environment.agents) * self.map_tries}\r", end='', flush=True)
                data = (i, j, self.environment.agents[i], start_tracks[j], starts[j])
                data_to_append.append(data)
        self.agents_feed.extend(data_to_append)
        self.waiting_for_agents.value = False

        start_time = time.time()
        time_per_unit = 0  
        smoothing_factor = 0.1
        while len(self.agents_feed) != 0 or any(self.working):
            remaining = len(self.agents_feed)
            total = len(self.environment.agents) * self.map_tries
            time_per_unit = (time.time() - start_time) / (total - remaining + 1) + smoothing_factor * time_per_unit
            eta = time_per_unit * remaining
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta)) + "             "
            print_progress(total, total - remaining, (total - remaining) / total, 30, eta_str)
            time.sleep(0.1) # - Wait for all agents to be fed to processes

        for i in range(len(self.environment.agents)):
            self.environment.agents[i].car.lap_time = self.lap_times[i]
            self.environment.agents[i].car.laps = self.laps[i]
            self.environment.agents[i].car.score = self.scores[i]

        self.environment.next_generation(self)

        print(f" - Moving to generation: {self.environment.generation}, best lap: {self.environment.previous_best_lap}, best completion: {(self.environment.previous_best_score/ (score_multiplier * self.map_tries) * 100):0.2f}%, max laps finished: {max(self.laps)}       ")

        for i in range(len(self.environment.agents)):
            self.scores[i] = 0
            self.lap_times[i] = 0
            self.laps[i] = 0
    
    def load_single_track(self):
        self.tracks = {}
        self.start_positions = {}
        self.center_lines ={}
        self.real_starts = {}
        track_name = self.options['track_name']
        data = self.load_track(track_name)

        self.tracks[track_name] = np.array(data['track'])
        self.track = self.tracks[track_name]
        self.center_lines[track_name] = data['center_line']
        self.start_positions[track_name] = data['start_positions']
        self.real_starts[track_name] = data['real_start']

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
            self.start_dir = start_dir[self.track_name]

        if self.player == 3:
            self.map_tries = 1
            self.tracks = {}
            self.tracks[self.track_name] = self.track

        print(f" - Loaded track: {track_name}")

    def GenerateTrainingStarts(self):
        start_tracks = []
        starts = []

        if self.player != 3:
            chosen_tracks = []
            count = 0
            while len(chosen_tracks) < real_starts_num and count < 100:
                track = random.choice(list(self.tracks.keys()))
                if track not in chosen_tracks:
                    start_tracks.append(track)
                    start_x = self.real_starts[track][0][0]
                    start_y = self.real_starts[track][0][1]
                    starts.append([[start_x, start_y], self.real_starts[track][1]])
                    chosen_tracks.append(track)
                count += 1
            while len(starts) < self.map_tries:
                track = random.choice(list(self.tracks.keys()))
                start_tracks.append(track)
                rand_start = random.choice(self.start_positions[track])
                starts.append(rand_start)
        else:
            start_tracks = [self.track_name]
            start_x = self.real_starts[self.track_name][0][0]
            start_y = self.real_starts[self.track_name][0][1]
            starts = [[[start_x, start_y], start_dir[self.track_name]]]

        for i in range(len(start_tracks)):
            assert(self.tracks[start_tracks[i]][starts[i][0][0], starts[i][0][1]] == 10)
        return starts, start_tracks

    def load_all_tracks(self):
        self.start_positions = Manager().dict()
        self.tracks = Manager().dict()
        self.center_lines = {}
        self.real_starts = Manager().dict()
        for file in os.listdir("./data/tracks"):
            if file.endswith(".png") and not file.endswith("_surface.png"):
                print(f" - Loading track: {file[:-4]}         \r", end='', flush=True)
                folder_path = f"./data/per_track/{file[:-4]}/trained"
                os.makedirs(folder_path, exist_ok=True)

                if self.player == 8 and file[:-4] != self.options['track_name']: continue
                self.track_name = file[:-4]
                data = self.load_track(file[:-4])
                
                self.tracks[file[:-4]] = np.array(data['track'])
                self.track = self.tracks[file[:-4]]
                self.center_lines[file[:-4]] = data['center_line']
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
            self.start_dir = start_dir[self.track_name]
        
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
                            self.board[y][x] = 100 + len(start_positions)
                            if len(start_positions) == 1:
                                real_start = [[x, y], start_dir[track_name]]
            self.track=np.array(self.board)
            
            print(" - Finding center line")
            self.find_center_line()
            center_line_coords = np.argwhere(self.track == 10).tolist()
            print(" - Generating new starts")
            starts = get_new_starts(self.track, 1000)
            real_start_x = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[0]
            real_start_y = get_nearest_centerline(self.track, real_start[0][0], real_start[0][1])[1]
            real_start = [[real_start_y, real_start_x], real_start[1]]
            print(" - Generating center line inputs")
            center_line = self.init_center_line()
            assert(real_start != None)

            data = {
                "track": self.track,
                "start_positions": starts,
                "center_line": center_line,
                "real_start": real_start
            }
            
            np.save("./data/tracks/" + track_name, data)
            print(" - Smoothening track")
            smoothen(track_name)
            print(f" - Generated track, calculated a ppm of: {str(len(center_line_coords) / real_track_lengths[track_name])}")
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

    def init_center_line(self):
        obtained_center_line = {}
        car = Car(self.track, (0,0), 0, self.track_name)
        track_pos = []
        if self.player == 5: return
        track_pos = np.argwhere(self.track == 10).tolist()
        done = 0
        for pos in track_pos:
            print(f" - Calculating center line inputs: {(done/len(track_pos) * 100):0.1f}%", end="\r")
            done += 1
            car.y = int(pos[0])
            car.x = int(pos[1])
            for i in range(8):
                car.direction = i * 45
                index = f"{int(car.y)}{int(car.x)}{int(i * 45)}"
                offset = offsets[directions.tolist().index(i*45)]
                if self.track[car.y + offset[1], car.x + offset[0]] == 10:
                    try:
                        obtained_center_line[index] = car.CalculateCenterlineInputs(int(car.x), int(car.y), i * 45)
                    except Exception as e:
                        if self.debug: print("Error calculating center line input" + str(e))
                        pass
        return obtained_center_line
    
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
        for track_name in self.track_names:
            track = self.tracks[track_name]
            pos, direction = random.choice(self.start_positions[track_name])
            for agent in self.environment.agents:
                agent.car.track = track
                agent.car.track_name = track_name
                agent.car.x = pos[1]
                agent.car.y = pos[0]
                agent.car.direction = direction
            start_time = time.time()
            for agent in self.environment.agents:
                for i in range(perft_ticks):
                    agent.Tick(i, self)
            tick_time += time.time() - start_time
        score = tick_time / len(self.track_names) / len(self.environment.agents) / perft_ticks * 1000
        self.totalScore += score
        self.runs += 1
        self.environment = Environment(self.options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)
        
