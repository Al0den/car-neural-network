import numpy as np
import random
import time
import sys
import re
import shutil
import json

from PIL import Image

from precomputed import next_brake_speed_after_t, four_wide_offsets
from settings import *
from termcolor import colored

def calculate_distance(coords1, coords2):
    return np.sqrt(pow(coords1[0] - coords2[0], 2) + pow(coords1[1] - coords2[1], 2))

def is_color_within_margin(color, target_color, margin):
    return all(abs(a - b) <= margin for a, b in zip(color, target_color))

def copy_network(network):
    copied_network = []
    for layer_weights in network:
        copied_layer_weights = np.copy(layer_weights)
        copied_network.append(copied_layer_weights)
    return copied_network

a = 336.02
b = 4.81
c = 2.08362

def speed_after_t(t):
    return a-a/(1+pow((t/b), c))

def get_current_t(speed): 
    t = b * pow(-a/(speed - a + 0.01) - 1, 1/c)
    return t

def next_speed(current_speed, speed_pre_calc=[]):
    try:
        return speed_pre_calc[int(current_speed * 100)]
    except:
        if current_speed < 5:
            return current_speed + 1
        current_speed = int(current_speed * 100) / 100
        current_t = get_current_t(current_speed)
        next_t = current_t + delta_t
        new_speed = min(max_speed - 1, np.real(speed_after_t(next_t)))

        return new_speed

def new_brake_speed(current_speed):
    applied_speed = min(current_speed, max_speed)
    diff = current_speed - applied_speed
    return next_brake_speed_after_t[min(int(applied_speed * 10)/10, 329.9)] + diff

def angle_distance(angle1, angle2):
    val = abs(angle1 - angle2) % 360
    if val > 180: return 360 - val
    return val

def get_new_starts(track, n, turn_intensity, track_name):
    start_positions = []

    positions = np.argwhere(track == 10).tolist()
  
    print(f" - Found {len(positions)} potential starts")
    positions.sort(key=lambda pos: turn_intensity[pos[0], pos[1]])

    # Calculate the number of positions to keep (85% of total)
    num_positions_to_keep = int(len(positions) * 0.85)

    # Select the top 80% of positions
    positions = positions[:num_positions_to_keep]
    print(" - Found", len(positions), "valid start positions")
    if not positions: return []

    with open("./src/config.json", 'r') as f:
        config = json.load(f)

    end_pos_x, end_pos_y = config.get("end_pos").get(track_name)

    for _ in range(n):
        random.shuffle(positions)
        chosen_pos, chosen_dir = None, None
        while chosen_pos == None:
            pos = random.choice(positions)
            if calculate_distance(pos, (end_pos_x, end_pos_y)) < 200:
                continue
            pos_x = pos[1]
            pos_y = pos[0]
            angle_to_end = np.degrees(np.arctan2((pos_y - end_pos_y), pos_x - end_pos_x))

            potential_directions = []
            best_dir = None
            best_angle_dist = None
            for offset in four_wide_offsets:
                new_pos = (pos[1] + offset[0], pos[0] + offset[1]) #(x,y)
                if track[new_pos[1], new_pos[0]] == 10:
                    start_angle = np.degrees(np.arctan2(offset[1], offset[0]))
                    potential_directions.append(offset)
                    if best_angle_dist is None or abs(angle_distance(start_angle, angle_to_end)) < best_angle_dist:
                        best_angle_dist = abs(angle_distance(start_angle, angle_to_end))
                        best_dir = offset
            if not potential_directions: continue
            if len(potential_directions) != 2: continue
            chosen_pos = pos
            chosen_dir = best_dir
        start_positions.append((chosen_pos, np.degrees(np.arctan2(-chosen_dir[1], chosen_dir[0]))))
    return start_positions

def angle_range_180(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def print_progress(total, agent, progress,width, eta=None):
    bar_width = int(width * progress)
    eta_str = f"ETA: {eta}" if eta is not None else ""
    print(f"Agent: {agent}/{int(total)} | [{'█' * bar_width}{' ' * (width - bar_width)}] {progress * 100:.1f}% - {eta_str} \r", end='', flush=True)

def update_progress(remaining, total, progress_width=30):
    start_time = time.time()
    time_per_unit = 0  
    smoothing_factor = 0.2 

    while remaining > 0:
        progress = 1 - remaining / total

        elapsed_time = time.time() - start_time
        time_per_unit = (1 - smoothing_factor) * elapsed_time / (total - remaining) + smoothing_factor * time_per_unit

        units_remaining = len(remaining) - (total - remaining) - 1
        eta_seconds = units_remaining * time_per_unit
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        try:
            print_progress(total, total - remaining, progress, progress_width, eta_str)
        except IndexError:
            pass
        time.sleep(0.1) 

def copy_car(car, new_car):
    new_car.x = car.x
    new_car.y = car.y
    new_car.direction = car.direction
    new_car.speed = car.speed
    new_car.acceleration = car.acceleration
    new_car.brake = car.brake
    new_car.steer = car.steer
    new_car.lap_times = car.lap_times
    new_car.lap_time = car.lap_time
    new_car.score = car.score
    new_car.died = car.died
    return new_car

def interpolate_color(index):
    if index < 0 or index > 1:
        raise ValueError("Index must be between 0 and 1")

    green = (0, 255, 0)
    red = (255, 0, 0)

    r = int((1 - index) * green[0] + index * red[0])
    g = int((1 - index) * green[1] + index * red[1])
    b = int((1 - index) * green[2] + index * red[2])

    return (r, g, b, 255)

def SaveOptimalLine(track_matrix, track_name, generation):
    gray = (0, 0, 0)
    red = (255, 0, 0)

    color_map = {
        1: gray,
        2: gray,
        3: red,
        10: gray,
    }

    track_matrix = np.array(track_matrix)
    height, width = track_matrix.shape
   
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
    for key, color in color_map.items():
        rgba_array[track_matrix == key] = color + (255,)


    speed_indices = np.where((track_matrix <= 370) & (track_matrix > 10))
    for i, j in zip(*speed_indices):
        speed = (track_matrix[i, j] - 10) / 360  # Normalize speed from 1 (green) to 0 (red)
        speed_color = interpolate_color(speed)

        for k in range(-1, 2):
            for l in range(-1, 2):
                if 0 <= i + k < height and 0 <= j + l < width:
                    rgba_array[i + k, j + l] = speed_color
    EditSurfaceImage(track_name, rgba_array)
    image = Image.fromarray(rgba_array)
    image.save(f"./data/per_track/{track_name}/generated_{generation}.png")

def EditSurfaceImage(track, array):
    path = f"./data/tracks/{track}_surface.png"
    image = Image.open(path)
    # Get image as numpy array
    surface = np.array(image)


    # Put all non black pixels from array into the surface

    surface[array != 0] = array[array != 0]
    # Save the image
    image = Image.fromarray(surface)
    image.save(f"./data/tracks/{track}_surface_path.png")
    
def SaveAgentsSpeedGraph(speeds, throttle, brake, generation, track_name, steer):
    import matplotlib.pyplot as plt
    speeds.pop()
    throttle.pop()
    time = [i/60 for i in range(len(speeds))]
    brake = [i * -1 for i in brake]
    _, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot the first subplot (Speed)
    axs[0].plot(time, speeds)
    axs[0].set_ylabel('Speed')
    # Plot the second subplot (Throttle and Brake)
    axs[1].plot(time, throttle, label='Throttle')
    time = [i/60 for i in range(len(brake))]
    axs[1].plot(time, brake, label='Brake')
    axs[1].axhline(y=0, color='r', linestyle='--')
    axs[1].set_ylabel('Throttle/Brake')
    axs[1].legend()
    time = [i/60 for i in range(len(steer))]
    # Plot the third subplot (Steer)
    axs[2].plot(time, steer)
    axs[2].set_ylabel('Steer')
    axs[2].set_xlabel('Time (seconds)')
    axs[2].axhline(y=0, color='r', linestyle='--')

    plt.savefig(f"./data/per_track/{track_name}/speeds_{generation}.png")

def InitialiseDisplay(game, game_options, pygame):
    from render import Render
    pygame.init()
    clock = pygame.time.Clock()
    game.screen = pygame.display.set_mode((game_options['screen_width'], game_options['screen_height']), pygame.RESIZABLE | pygame.SRCALPHA)
    game.render = Render(game.screen, game_options, game)
    game.clock = clock
    pygame.display.set_caption("Car Game")
    last_render = time.time()

    return clock, last_render

def get_nearest_centerline(track, x, y):
    distance = 1
    for _ in range(100):
        for i in range(-distance, distance):
            for j in range(-distance, distance + 1):
                if track[y + i, x + j] == 10:
                    return x+j, y+i
        distance += 1
    print("Could find the nearest track, aborting")
    sys.exit()

def get_terminal_width():
    columns, _ = shutil.get_terminal_size()
    return columns

def update_terminal(game, total_agents, alive_agents, tot_ticks, input_percentage, metal_percentage, tick_percentage, TPS, RTS, generation, min_ticks, max_ticks, max_alive, min_alive, human_formatted, ts, tc, working, issues):
    terminal_width = get_terminal_width()
    
    generation_line = colored(f"Generation: {generation}", attrs=['bold'])

    progress = int(alive_agents / total_agents * (terminal_width * 1/2))
    progress_bar = f"Progress: [{colored('#' * progress, 'red')}{colored('.' * (int(terminal_width * 1/2) - progress), 'green')}] {alive_agents}/{total_agents}"
    def worker_color(worker_num):
        match worker_num:
            case 0: return 'red'
            case 1: return 'yellow'
            case 2: return 'green'
            case 3: return 'blue'
            case 4: return 'red'
    working_bar = ""
    for worker in working:
        working_bar += colored('█ ', worker_color(worker))

    def colored_length(text):
        # Function to calculate the length of the colored text
        return len(re.sub(r'\033\[[0-9;]+m', '', text))

    # Adjusted the formatting to consider colored text length
    info_line_1 = f"Input Percentage: {input_percentage:<5}% | Metal Percentage: {metal_percentage:<5}% | Tick Percentage: {tick_percentage:<5}%"
    info_line_2 = f"Total Ticks: {tot_ticks} | Ticks Per Second: {TPS} | Real Time Speed: {RTS}"
    info_line_3 = f"Min Ticks: {min_ticks} | Max Ticks: {max_ticks} | Min Alive: {min_alive} | Max Alive: {max_alive}"

    time_left_line = f"Time Spent: {human_formatted} - Ticks: {int(tot_ticks/game.options['cores'])}"

    tc_color = 'green' if tc >= 0 else 'red'
    ts_color = 'green' if ts <= 0 else 'red'

    # Adjusted the formatting to consider colored text length
    trajectory_line_1 = f"Lap Improvement: {colored(ts, ts_color)} | Score Improvement: {colored(tc, tc_color)} | Issues: {issues}"
    
    # Convert lap time to m:ss
    display_lap_time = time.strftime("%M:%S", time.gmtime(game.environment.previous_best_lap/60))

    if game.player == 3:
        target_lap_time = game.config.get("quali_laps").get(game.track_name)
        delta = round(game.environment.previous_best_lap/60 - target_lap_time, 2)
        delta_display = f"{delta}" if delta < 0 else f"+{delta}"
        results_line = f"Previous Completion: {game.environment.previous_best_score/(score_multiplier * game.map_tries) * 100:0.2f}% | Best Lap Time: {display_lap_time} | Delta: {delta_display}"
    else:
        results_line = f"Previous Completion: {game.environment.previous_best_score/(score_multiplier * game.map_tries) * 100:0.2f}% | Best Lap Time: {display_lap_time}"

    print(f"\033c\n{generation_line:^{terminal_width - colored_length(generation_line) + len(generation_line)}}\n\n"
          f"{progress_bar:^{terminal_width - colored_length(progress_bar) + len(progress_bar)}}\n\n"
          f"{info_line_1:^{terminal_width - colored_length(info_line_1) + len(info_line_1)}}\n\n"
          f"{info_line_2:^{terminal_width - colored_length(info_line_2) + len(info_line_2)}}\n\n"
          f"{info_line_3:^{terminal_width - colored_length(info_line_3) + len(info_line_3)}}\n\n"
          f"{trajectory_line_1:^{terminal_width - colored_length(trajectory_line_1) + len(trajectory_line_1)}}\n\n"
          f"{results_line:^{terminal_width - colored_length(results_line) + len(results_line)}}\n\n"
          f"{time_left_line:^{terminal_width - colored_length(time_left_line) + len(time_left_line)}}\n\n"
          f"{working_bar:^{terminal_width - colored_length(working_bar) + len(working_bar)}}\n")
