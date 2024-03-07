import numpy as np
import random
import time
import sys

from PIL import Image

from precomputed import next_brake_speed_after_t, four_wide_offsets
from settings import * 

def calculate_distance(coords1, coords2):
    return np.sqrt(pow(coords1[0] - coords2[0], 2) +pow(coords1[1] - coords2[1], 2))

def is_color_within_margin(color, target_color, margin):
    return all(abs(a - b) <= margin for a, b in zip(color, target_color))

def speed_after_t(t):
    return -max_speed * np.exp((-t)/4.5) + max_speed

def copy_network(network):
    copied_network = []
    for layer_weights in network:
        copied_layer_weights = np.copy(layer_weights)
        copied_network.append(copied_layer_weights)
    return copied_network

def next_speed(current_speed):
    current_t = -4.5 * np.log(1-(current_speed/max_speed))
    next_t = current_t + delta_t
    return speed_after_t(next_t)

def new_brake_speed(current_speed):
    applied_speed = min(current_speed, max_speed)
    diff = current_speed - applied_speed
    return next_brake_speed_after_t[min(int(applied_speed * 10)/10, 329.9)] + diff

def angle_distance(angle1, angle2):
    val = abs(angle1 - angle2) % 360
    if val > 180: return 360 - val
    return val

def get_new_starts(track, n, turn_intensity):
    start_positions = []

    positions = np.argwhere(track == 10).tolist()
    turn_intensity_threshold = np.mean(turn_intensity) * 10
    print(f" - Found {len(positions)} potential starts")
    positions = [pos for pos in positions if turn_intensity[pos[0], pos[1]] < turn_intensity_threshold]
    print(" - Found", len(positions), "valid start positions")
    if not positions: return []

    for _ in range(n):
        random.shuffle(positions)
        chosen_pos, chosen_dir = None, None
        while chosen_pos == None:
            pos = random.choice(positions)
            potential_directions = []
            for offset in four_wide_offsets:
                new_pos = (pos[1] + offset[0], pos[0] + offset[1])
                if track[new_pos[1], new_pos[0]] == 10:
                    potential_directions.append(offset)
            if not potential_directions: continue
            if len(potential_directions) != 2: continue
            chosen_pos = pos
            chosen_dir = random.choice(potential_directions)
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

    image = Image.fromarray(rgba_array)
    image.save(f"./data/per_track/{track_name}/generated_{generation}.png")

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