import numpy as np

points_offset = [0, -20, 20, 55, -55, 89, -89]

# - Neural network settings
center_line_input = True
state_space_size = len(points_offset) + 4
first_layer_size_coeff = 1.3
num_hidden_layers = 3
action_space_size = 2
activation_function = np.tanh
map_tries = 10
real_starts_num = 3

travel_distances_centerlines = [1, 10, 20, 30, 40, 50, 60, 80, 100, 133, 166, 200, 233, 266, 300, 333, 366, 400, 433, 466, 500, 533, 566, 600]

max_points_distance = 200
max_center_line_distance = 70

state_space_size += len(travel_distances_centerlines)

first_layer_size = int(state_space_size * first_layer_size_coeff)

# - Parameters
point_search_jump = 25.0
perft_ticks = 5
perft_duration = 20

# - Game settings
god = False
debug = False
random_start_position = True
delta_t = 1/60
pre_load = True
min_checkpoint_distance = 150
max_time_on_checkpoint = 200
base_game_speed = 60

# - Simulation settings
car_length = 5.23
car_width = 1.8
max_speed = 340
turn_coeff = 4.5
drag_coeff = 0.881
reference_area = 1.7
lift_coeff = -1.247
car_mass = 733
gravity = 9.81
brake_increment = 1/10
acceleration_increment = 1/10
steer_increment = 1/10

# - Evolution settings
mutation_rates = [0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3]

only_mutate_rate = 0.7
cross_over_rate = 0.25
mutate_cross_over_rate = 0.7
simple_cross_over_rate = 0.3
random_agent_rate = 0.05

previous_ratio = 0.05
no_lap_value = 1000000
score_multiplier = 10000

mutation_strenght = 0.1

max_ticks_before_kill = 30000
min_speed = 10
base_score_per_checkpoint = 50

# - Render Options
default_width = 960
default_height = 510

