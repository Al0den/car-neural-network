import os
import numpy as np
import threading
import random
import time

from multiprocessing import Process, Array, Manager, Value
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from environment import Environment
from car import Car
from utils import is_color_within_margin, calculate_distance, get_new_starts, update_progress, get_nearest_centerline
from agent import Agent
from settings import *
from precomputed import start_dir, offsets, directions
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
        self.running = True # If the game needs to be stopped, set this to False to kill main loop

        self.map_tries = map_tries # - Number of total tries
        self.mutation_strength = game_options['mutation_strength']

        self.screen = None # Display parameter
        self.render = None # Display parameter
        self.clock = None # Display parameter
        self.visual = True # - Display parameter
        self.debug = debug # - Display parameter
        self.last_keys_update = 0 # - Time since last key click, Display parameter
        self.track_name = game_options['track_name'] # - Track name

        if self.player != 0:
            self.load_all_tracks()
        else:
            self.load_single_track()
            
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
            start_x = self.real_starts[self.track_name][0][0]
            start_y = self.real_starts[self.track_name][0][1]
            real_start_x = get_nearest_centerline(self.tracks[self.track_name], start_x, start_y)[0]
            real_start_y = get_nearest_centerline(self.tracks[self.track_name], start_x, start_y)[1]
            self.environment.agents[0].car.x = real_start_x
            self.environment.agents[0].car.y = real_start_y
            assert(self.tracks[self.track_name][real_start_y, real_start_x] == 10)
            self.environment.agents[0].car.direction = self.real_starts[self.track_name][1]

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
        elif self.player == 10:
            print(" - Starting performance test...")
            self.totalScore = 0
            self.runs = 0
            self.prev_update = time.time()
            self.start_time = time.time()
    def tick(self):
        if self.player == 0:
            self.ticks += 1
            if self.ticks % 100 == 0 and debug:
                print("Ticks: " + str(self.ticks))
            self.car.tick(self)
            if self.car.died == True:
                self.restart = True
                self.car.died = False
                if debug: print("Car died, restarting")
        elif self.player == 1 or self.player == 3:
            self.train_agents()
        elif self.player == 4:
            self.environment.agents[0].tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 5:
            import pygame
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d] or keys[pygame.K_h]:
                self.environment.agents[0].car.applyPlayerInputs()
                self.environment.agents[0].car.updateCar()
                self.environment.agents[0].car.checkCollisions(self.ticks)
            else:
                self.environment.agents[0].tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 7:
            self.environment.agents[0].tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 8:
            for agent in self.environment.agents:
                agent.tick(self.ticks, self)
            self.ticks += 1
        elif self.player == 10:
            self.performanceTest()
            # If more than 2s has passed since last update, update the screen
            if time.time() - self.prev_update > 2:
                self.prev_update = time.time()
                print(f"Average time/tick: {self.totalScore / self.runs:.5f}ms, ran {self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks} ticks")
            if time.time() - self.start_time > perft_duration:
                print(f"Average tps: {(self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks / perft_duration):.2f}, real speed x{(self.runs * len(self.track_names) * self.options['environment']['num_agents'] * perft_ticks / (perft_duration * 1/delta_t)):.2f}")
                self.running = False
    def train_agents_process(self, agents, remaining, laps, lap_times, scores, starts, index_lock, current_index, start_tracks):
        local_scores = [0] * len(agents)
        local_lap_times = [0] * len(agents) * self.map_tries
        while current_index.value < len(remaining):
            with index_lock:
                num = current_index.value
                current_index.value += 1
            if num >= len(remaining):
                break
            agent_index = remaining[num]
            try_number = num % self.map_tries
            agent = agents[agent_index]
            if agent.car.died:
                agent.car.died = False
            agent.car.lap_time = 0
            agent.car.track = self.tracks[start_tracks[try_number]]
            agent.car.track_name = start_tracks[try_number]
            agent.car.x = starts[try_number][0][1]
            agent.car.y = starts[try_number][0][0]
            agent.car.direction = starts[try_number][1]
            agent.car.start_direction = starts[try_number][1]
            agent.car.start_x = starts[try_number][0][1]
            agent.car.start_y = starts[try_number][0][0]
            ticks = 0
            while not agent.car.died:
                agent.tick(ticks, self, self.player)
                ticks += 1
            if agent.car.lap_time != 0:
                with laps.get_lock():
                    laps[agent_index] += 1
                local_lap_times[agent_index * self.map_tries + try_number] = int(agent.car.lap_time)
                local_scores[agent_index] += 1 * score_multiplier
            else:
                local_lap_times[agent_index * self.map_tries + try_number] = 0
                local_scores[agent_index] += min(1 * score_multiplier, int((agent.car.calculateScore() * score_multiplier)))
        with lap_times.get_lock():
            for i in range(len(lap_times)):
                lap_times[i] += local_lap_times[i]
        with scores.get_lock():
            for i in range(len(scores)):
                scores[i] += local_scores[i]
    
    def train_agents(self):
        num_agents = len(self.environment.agents)
        num_processes = self.options['cores']

        processes = []
        scores = Array('i', [0] * num_agents)
        lap_times = Array('i', [0] * self.map_tries * num_agents)
        laps = Array('i', [0] * num_agents)
        remaining = Array('i', [0] * self.map_tries * num_agents)
        index_lock = Manager().Lock()
        index = Value('i', 0)

        for i in range(num_agents):
            for j in range(self.map_tries):
                remaining[i * self.map_tries + j] = i
        start_tracks = [random.choice(list(self.tracks.keys())) for _ in range(self.map_tries - real_starts_num)]
        starts = []
        for track in start_tracks:
            new_start = random.choice(self.start_positions[track])
            starts.append(new_start)

        if self.player != 3:
            chosen_tracks = []
            count = 0
            while len(chosen_tracks) < real_starts_num and count < 100:
                track = random.choice(list(self.tracks.keys()))
                if track not in chosen_tracks:
                    start_tracks.append(track)
                    start_x = self.real_starts[track][0][0]
                    start_y = self.real_starts[track][0][1]
                    real_start_x = get_nearest_centerline(self.tracks[track], start_x, start_y)[0]
                    real_start_y = get_nearest_centerline(self.tracks[track], start_x, start_y)[1]
                    starts.append([[real_start_y, real_start_x], self.real_starts[track][1]])
                    chosen_tracks.append(track)
                count += 1
            for i in range(real_starts_num - len(chosen_tracks)):
                track = random.choice(list(self.tracks.keys()))
                start_tracks.append(track)
                starts.append(self.real_starts[track])
        else:
            start_tracks = [self.track_name]
            start_x = self.real_starts[self.track_name][0][0]
            start_y = self.real_starts[self.track_name][0][1]
            real_start_x = get_nearest_centerline(self.tracks[self.track_name], start_x, start_y)[0]
            real_start_y = get_nearest_centerline(self.tracks[self.track_name], start_x, start_y)[1]
            starts = [[[real_start_y, real_start_x], self.real_starts[self.track_name][1]]]
        progress_width = 30

        for i in range(len(start_tracks)):
            assert(self.tracks[start_tracks[i]][starts[i][0][0], starts[i][0][1]] == 10)

        for i in range(num_processes):
            process = Process(target=self.train_agents_process, args=(self.environment.agents, remaining, laps, lap_times, scores, starts, index_lock, index, start_tracks))
            processes.append(process)

        progress_thread = threading.Thread(target=update_progress, args=(index, remaining, self, progress_width))
        progress_thread.daemon = True
        progress_thread.start()

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        index.value = len(remaining)

        lap_times = np.array(lap_times).reshape((num_agents, self.map_tries))

        for i in range(num_agents):
            self.environment.agents[i].car.score = scores[i]
            self.environment.agents[i].car.laps = laps[i]
            self.environment.agents[i].car.lap_times = lap_times[i]

        self.environment.next_generation(self)

        print(f"Moving to generation: {self.environment.generation}, best lap: {self.environment.previous_best_lap}, best completion: {(self.environment.previous_best_score/ (score_multiplier * self.map_tries) * 100):0.2f}%, max laps finished: {max(laps)}    ")
    def load_single_track(self):
        self.tracks = {}
        self.start_positions = {}
        self.center_lines = {}
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

        print(f" - Loaded track: {track_name}")
    def load_all_tracks(self):
        self.start_positions = {}
        self.tracks = {}
        self.center_lines = {}
        self.real_starts = {}
        print(" * Loading tracks...")
        for file in os.listdir("./data/tracks"):
            if file.endswith(".png") and not file.endswith("_surface.png"):
                if debug: print(f" - Loading track: {file[:-4]}")
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
        
        self.agent = False

        if self.player == 3:
            self.map_tries = 1
            self.tracks = {}
            self.tracks[self.track_name] = self.track

        if self.player in [4, 8]:
            self.tracks = {}
            self.tracks[self.track_name] = self.track
            self.track_name = self.options['track_name']

        print(f" * Loaded tracks: {str(', '.join(list(self.tracks.keys())))}")

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
                                real_start = ((x, y), start_dir[track_name])
            self.track=np.array(self.board)

            print(" - Finding center line")
            self.find_center_line()
            print(" - Generating new starts")
            starts = get_new_starts(self.track, 1000)
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
            print(f" - Generated track, center-line: {str(len(center_line))}")
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
                        obtained_center_line[index] = car.center_line_3_input(int(car.x), int(car.y), i * 45)
                    except Exception as e:
                        if debug: print("Error calculating center line input" + str(e))
                        pass
        print()
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
                    agent.tick(i, self)
            tick_time += time.time() - start_time
        score = tick_time / len(self.track_names) / len(self.environment.agents) / perft_ticks * 1000
        self.totalScore += score
        self.runs += 1
        self.environment = Environment(self.options['environment'], self.track, self.player, self.start_pos, self.start_dir, self.track_name)
        
